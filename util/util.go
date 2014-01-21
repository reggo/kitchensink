// Set of utility functions for the kitchensinks algorithm. The point is to
// provide public routines for advanced users while keeping the main
// package simple. These functions do not provide error checking, so use
// with caution
package util

import (
	"math"
	//"runtime"
	//"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"

	//"github.com/reggo/common"
	//"github.com/reggo/loss"
	//"github.com/reggo/regularize"
)

const BatchSize = 100 // Number of predictions to do at once when predicting in parallel

// ComputeZ computes the value of z with the given feature vector and b value.
// Sqrt2OverD = math.Sqrt(2.0 / len(nFeatures))
func ComputeZA(input, feature []float64, b float64, sqrt2OverD float64) float64 {
	dot := floats.Dot(input, feature)
	return sqrt2OverD * (math.Cos(dot + b))
}

// Predict predicts the output at the input with the given features and feature weights.
// The output value is put in-place into output.
func Predict(input []float64, features *mat64.Dense, b []float64, featureWeights *mat64.Dense, output []float64) {
	for i := range output {
		output[i] = 0
	}

	nFeatures, outputDim := features.Dims()

	sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))
	//for i, feature := range features {
	for i := 0; i < nFeatures; i++ {
		z := ComputeZA(input, features.RowView(i), b[i], sqrt2OverD)
		for j := 0; j < outputDim; j++ {
			output[j] += z * featureWeights.At(i, j)
		}
	}
}

/*

// SequentialPredict predicts the outputs at a slice of inputs sequentially
func SequentialPredict(inputs [][]float64, features *mat64.Dense, b []float64, featureWeights [][]float64, outputs [][]float64) {
	for i, input := range inputs {
		Predict(input, features, b, featureWeights, outputs[i])
	}
}

func getBatchSize(nSamples int) int {
	nCpu := runtime.GOMAXPROCS(0)
	samplesPerProc := nSamples/nCpu + 1
	batchSize := BatchSize
	if samplesPerProc < batchSize {
		batchSize = samplesPerProc
	}
	return batchSize
}

// ParallelPredict predicts the outputs at a slice of inputs in parallel
func ParallelPredict(inputs [][]float64, features *mat64.Dense, b []float64, featureWeights [][]float64, outputs [][]float64) {

	nInputs := len(inputs)
	batchSize := getBatchSize(nInputs)
	counter := 0
	wg := &sync.WaitGroup{}
	for counter != nInputs {
		oldCounter := counter
		counter += batchSize
		if counter > nInputs {
			counter = nInputs
		}
		wg.Add(1)
		go func(minInd int, maxInd int) {
			SequentialPredict(inputs[minInd:maxInd], features, b, featureWeights, outputs[minInd:maxInd])
			wg.Done()
		}(oldCounter, counter)
	}
	wg.Wait()
}

// PredictWithZ predicts the output with the given weights and pre-computed
// z values. The predicted output is put in-place to output.
func PredictWithZ(z []float64, featureWeights [][]float64, output []float64) {
	for i := range output {
		output[i] = 0
	}
	for j, zval := range z {
		for i, weight := range featureWeights[j] {
			output[i] += weight * zval
		}
	}
}

func Deriv(z []float64, featureWeights [][]float64, dLossDPred []float64, dLossDWeight [][]float64) {
	// dLossDWeight_ij = \sum_k dLoss/dPred_k * dPred_k / dWeight_j

	// The prediction is just weights * z, so dPred_jDWeight_i = z_i
	for i, zVal := range z {
		for j := range dLossDWeight[i] {
			dLossDWeight[i][j] = zVal
		}
	}

	// dLossDWeight = dLossDPred * dPredDWeight
	for i := range dLossDWeight {
		for j, deriv := range dLossDPred {
			dLossDWeight[i][j] *= deriv
		}
	}
}

// PredLossDeriv predicts the value at an input with the given pre-computed features z.
// predOutput and dLossDPred are both temporary memory
func PredLossDeriv(z, trueOutput []float64, featureWeights *common.SlicedSlice, losser loss.DerivLosser, regularizer regularize.Regularizer,
	dLossDWeight *common.SlicedSlice, predOutput, dLossDPred []float64) (loss float64) {

	PredictWithZ(z, featureWeights.Mat, predOutput)

	loss = losser.LossAndDeriv(predOutput, trueOutput, dLossDPred)

	Deriv(z, featureWeights.Mat, dLossDPred, dLossDWeight.Mat)

	// Deriv has stored the values, now add the regularized weights
	loss += regularizer.LossAddDeriv(featureWeights.Slice, dLossDWeight.Slice)
	return
}

// Stores the derivative as a slice in dLossDWeight
func SeqLossDeriv(z, trueOutput [][]float64, featureWeights *common.SlicedSlice, losser loss.DerivLosser, regularizer regularize.Regularizer,
	dLossDWeight, dLossDWeightTmp *common.SlicedSlice, predOutput, dLossDPred []float64) (loss float64) {

	for i := range dLossDWeight.Slice {
		dLossDWeight.Slice[i] = 0
	}
	for i := range z {
		loss += PredLossDeriv(z[i], trueOutput[i], featureWeights, losser, regularizer, dLossDWeightTmp, predOutput, dLossDPred)
		for j := range dLossDWeight.Slice {
			dLossDWeight.Slice[j] += dLossDWeightTmp.Slice[j]
		}
	}
	return
}

type seqResult struct {
	dLossDWeight []float64
	loss         float64
}

// ParLossDeriv computes the loss and derivative with respect to the weights in parallel
func ParLossDeriv(z, trueOutput [][]float64, featureWeights *common.SlicedSlice, losser loss.DerivLosser, regularizer regularize.Regularizer) (loss float64, dLossDWeight *common.SlicedSlice) {

	nSamples := len(z)
	nFeatures := len(z[0])
	nOutputs := len(trueOutput[0])
	batchSize := getBatchSize(nSamples)
	dLossDWeight = common.NewSlicedSlice(nFeatures, nOutputs)

	c := make(chan seqResult, 4)
	counter := 0

	wg := &sync.WaitGroup{}
	for counter != nSamples {
		oldCounter := counter
		counter += batchSize
		if counter > nSamples {
			counter = nSamples
		}
		wg.Add(1)
		go func(minInd int, maxInd int) {
			predOutput := make([]float64, nOutputs)
			dLossDPred := make([]float64, nOutputs)
			dLossDWeightInner := common.NewSlicedSlice(nFeatures, nOutputs)
			dLossDWeightTmp := common.NewSlicedSlice(nFeatures, nOutputs)
			loss := SeqLossDeriv(z[minInd:maxInd], trueOutput[minInd:maxInd], featureWeights, losser, regularizer, dLossDWeightInner, dLossDWeightTmp, predOutput, dLossDPred)
			c <- seqResult{dLossDWeight: dLossDWeightInner.Slice, loss: loss}
			wg.Done()
		}(oldCounter, counter)
	}
	go func(wg *sync.WaitGroup) {
		wg.Wait()
		close(c)
	}(wg)
	loss = 0
	for s := range c {
		loss += s.loss
		for i, val := range s.dLossDWeight {
			dLossDWeight.Slice[i] += val
		}
	}
	return
}

*/
