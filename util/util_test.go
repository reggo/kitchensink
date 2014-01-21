package util

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/gonum/floats"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"
)

const (
	fdStep = 1e-6
	fdTol  = 1e-6
)

func TestComputeZ(t *testing.T) {
	for _, test := range []struct {
		x         []float64
		feature   []float64
		b         float64
		z         float64
		nFeatures float64
		name      string
	}{
		{
			name:      "General",
			x:         []float64{2.0, 1.0},
			feature:   []float64{8.1, 6.2},
			b:         0.8943,
			nFeatures: 50,
			z:         -0.07188374176,
		},
	} {
		z := ComputeZ(test.x, test.feature, test.b, math.Sqrt(2.0/test.nFeatures))
		if floats.EqualWithinAbsOrRel(z, test.z, 1e-14, 1e-14) {
			t.Errorf("z mismatch for case %v. %v expected, %v found", test.name, test.z, z)
		}
	}
}

func TestPredictWithZ(t *testing.T) {
	for _, test := range []struct {
		z              []float64
		featureWeights [][]float64
		output         []float64
		Name           string
	}{
		{
			Name: "General",
			z:    []float64{1, 2, 3},
			featureWeights: [][]float64{
				{3, 4},
				{1, 2},
				{0.5, 0.4},
			},
			output: []float64{6.5, 9.2},
		},
	} {
		zCopy := make([]float64, len(test.z))
		copy(zCopy, test.z)
		fWCopy := make([][]float64, len(test.featureWeights))
		for i := range test.featureWeights {
			fWCopy[i] = make([]float64, len(test.featureWeights[i]))
			copy(fWCopy[i], test.featureWeights[i])
		}
		output := make([]float64, len(test.output))

		PredictWithZ(zCopy, fWCopy, output)

		// Test that z wasn't changed
		if !floats.Equal(test.z, zCopy) {
			t.Errorf("z changed during call to PredictWithZ")
		}
		eq := true
		for i, weights := range test.featureWeights {
			if !floats.Equal(weights, fWCopy[i]) {
				eq = false
				break
			}
		}
		if !eq {
			t.Errorf("feature weights changed during call to PredictWithZ. Previously %v, now %v", test.featureWeights, fWCopy)
		}

		if !floats.EqualApprox(output, test.output, 1e-14) {
			t.Errorf("output doesn't match for test %v. Expected %v, found %v", test.Name, test.output, output)
		}
	}
}

func TestPredict(t *testing.T) {
	for _, test := range []struct {
		input          []float64
		features       [][]float64
		b              []float64
		featureWeights [][]float64
		Name           string
	}{
		{
			input: []float64{8, 9, 10},
			featureWeights: [][]float64{
				{8, 9},
				{0.4, 0.2},
				{9.8, 1.6},
				{-4, -8},
			},
			features: [][]float64{
				{0.9, 0.8, 0.7},
				{-0.7, 0.2, 15},
				{1.5, 7.8, -2.4},
				{9.7, 9.2, 1.2},
			},
			b:    []float64{0.7, 1.2, 0.2, 0.01234},
			Name: "General",
		},
	} {
		inputCopy := make([]float64, len(test.input))
		copy(inputCopy, test.input)

		fWCopy := make([][]float64, len(test.featureWeights))
		for i := range fWCopy {
			fWCopy[i] = make([]float64, len(test.featureWeights[i]))
			copy(fWCopy[i], test.featureWeights[i])
		}

		featuresCopy := make([][]float64, len(test.features))
		for i := range featuresCopy {
			featuresCopy[i] = make([]float64, len(test.features[i]))
			copy(featuresCopy[i], test.features[i])
		}

		bCopy := make([]float64, len(test.b))
		copy(bCopy, test.b)

		// This test assumes ComputeZ and PredictWithZ work
		nOutput := len(test.featureWeights[0])
		nFeatures := len(test.featureWeights)
		zOutput := make([]float64, nOutput)
		predOutput := make([]float64, nOutput)

		for i := range predOutput {
			predOutput[i] = rand.NormFloat64()
			zOutput[i] = rand.NormFloat64()
		}

		sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))

		z := make([]float64, nFeatures)
		for i := range z {
			z[i] = ComputeZ(test.input, test.features[i], test.b[i], sqrt2OverD)
		}
		PredictWithZ(z, test.featureWeights, zOutput)

		Predict(inputCopy, featuresCopy, bCopy, fWCopy, predOutput)

		// Check to make sure nothing changed
		if !floats.Equal(inputCopy, test.input) {
			t.Errorf("input has been modified")
		}
		if !floats.Equal(bCopy, test.b) {
			t.Errorf("b has been modified")
		}
		eq := true
		for i := range fWCopy {
			if !floats.Equal(fWCopy[i], test.featureWeights[i]) {
				eq = false
				break
			}
		}
		if !eq {
			t.Errorf("feature weights changed")
		}
		eq = true
		for i := range featuresCopy {
			if !floats.Equal(test.features[i], featuresCopy[i]) {
				eq = false
			}
		}
		if !eq {
			t.Errorf("features changed")
		}
		if !floats.EqualApprox(zOutput, predOutput, 1e-14) {
			t.Errorf("Prediction doesn't match for case %v. Expected %v, found %v", test.Name, zOutput, predOutput)
		}
	}
}

func randomSoS(outer, inner int) [][]float64 {
	s := make([][]float64, outer)
	for i := range s {
		s[i] = make([]float64, inner)
		for j := range s[i] {
			s[i][j] = rand.NormFloat64()
		}
	}
	return s
}

func TestSlicePredict(t *testing.T) {
	for _, test := range []struct {
		nFeatures int
		nOutput   int
		nInput    int
		nDim      int
		nCpu      int
		Name      string
	}{
		{
			Name:      "FullBatch",
			nFeatures: 8,
			nOutput:   3,
			nDim:      3,
			nCpu:      4,
			nInput:    BatchSize * 2 * 4,
		},
		{
			Name:      "FullBatchOffset",
			nFeatures: 8,
			nOutput:   3,
			nDim:      3,
			nCpu:      4,
			nInput:    BatchSize*2*4 + 9,
		},
		{
			Name:      "MoreThanCpuLessThanBatchOff",
			nFeatures: 8,
			nOutput:   3,
			nDim:      3,
			nCpu:      4,
			nInput:    11,
		},
		{
			Name:      "MoreThanCpuLessThanBatchExact",
			nFeatures: 8,
			nOutput:   3,
			nDim:      3,
			nCpu:      4,
			nInput:    16,
		},
	} {
		runtime.GOMAXPROCS(test.nCpu)
		nInput := test.nInput
		nDim := test.nDim
		nOutput := test.nOutput
		nFeatures := test.nFeatures

		inputs := randomSoS(nInput, nDim)
		features := randomSoS(nFeatures, nDim)
		featureWeights := randomSoS(nFeatures, nOutput)
		b := make([]float64, nFeatures)
		for i := range b {
			b[i] = rand.NormFloat64()
		}

		indPredOutput := randomSoS(nInput, nOutput)
		seqPredOutput := randomSoS(nInput, nOutput)
		parPredOutput := randomSoS(nInput, nOutput)

		for i := range inputs {
			Predict(inputs[i], features, b, featureWeights, indPredOutput[i])
		}
		SequentialPredict(inputs, features, b, featureWeights, seqPredOutput)
		ParallelPredict(inputs, features, b, featureWeights, parPredOutput)

		// Check that all the predictions match
		eq := true
		for i, out := range indPredOutput {
			if !floats.Equal(seqPredOutput[i], out) {
				eq = false
				break
			}
		}
		if !eq {
			t.Errorf("Sequential and individual don't match for case %v", test.Name)
			continue
		}

		eq = true
		for i, out := range indPredOutput {
			if !floats.Equal(parPredOutput[i], out) {
				eq = false
				break
			}
		}

		if !eq {
			t.Errorf("Parallel and individual don't match for case %v", test.Name)
			continue
		}
	}
}

func TestPredLossDeriv(t *testing.T) {
	//nInput := 3
	nFeatures := 10
	nOutput := 2

	z := make([]float64, nFeatures)
	for i := range z {
		z[i] = rand.NormFloat64() * 9.2
		//z[i] = 0
	}

	featureWeights := common.NewSlicedSlice(nFeatures, nOutput)
	for i := range featureWeights.Slice {
		featureWeights.Slice[i] = rand.NormFloat64() * 200.7
	}
	trueOutput := make([]float64, nOutput)
	for i := range trueOutput {
		trueOutput[i] = rand.NormFloat64()
	}
	losser := loss.ManhattanDistance{}
	regularizer := &regularize.TwoNorm{Gamma: 0.001}

	dLossDWeight := common.NewSlicedSlice(nFeatures, nOutput)
	predOutput := make([]float64, nOutput)
	dLossDPred := make([]float64, nOutput)

	loss := PredLossDeriv(z, trueOutput, featureWeights, losser, regularizer, dLossDWeight, predOutput, dLossDPred)

	predTmp := make([]float64, nOutput)

	// Check that loss matches
	PredictWithZ(z, featureWeights.Mat, predTmp)
	lossCheck := losser.Loss(predTmp, trueOutput)
	lossCheck += regularizer.Loss(featureWeights.Slice)

	if math.Abs(lossCheck-loss) > 1e-14 {
		t.Errorf("Loss mismatch")
	}

	// Test that derivative is true using finite difference
	dLossDWeightFd := common.NewSlicedSlice(nFeatures, nOutput)

	fDStep := 1e-6
	fDTol := 1e-6

	for i := range dLossDWeightFd.Slice {
		featureWeights.Slice[i] += fDStep
		PredictWithZ(z, featureWeights.Mat, predTmp)
		loss1 := losser.Loss(predTmp, trueOutput)
		loss1 += regularizer.Loss(featureWeights.Slice)
		featureWeights.Slice[i] -= 2 * fDStep
		PredictWithZ(z, featureWeights.Mat, predTmp)
		loss2 := losser.Loss(predTmp, trueOutput)
		loss2 += regularizer.Loss(featureWeights.Slice)
		featureWeights.Slice[i] += fDStep

		dLossDWeightFd.Slice[i] = (loss1 - loss2) / (2 * fDStep)
	}
	if !floats.EqualApprox(dLossDWeightFd.Slice, dLossDWeight.Slice, fDTol) {
		difference := common.NewSlicedSlice(nFeatures, nOutput)
		for i := range dLossDWeightFd.Slice {
			difference.Slice[i] = dLossDWeightFd.Slice[i] - dLossDWeight.Slice[i]
		}
		for i := range dLossDWeight.Mat {
			//fmt.Println(dLossDWeightFd.Mat[i], dLossDWeight.Mat[i], difference.Mat[i])
			fmt.Println(dLossDWeightFd.Mat[i])
			fmt.Println(dLossDWeight.Mat[i])
			fmt.Println(difference.Mat[i])
			fmt.Println()
		}
		t.Errorf("Derivative doesn't match")
	}
}

func TestSliceLossDeriv(t *testing.T) {
	for _, test := range []struct {
		nFeatures int
		nOutput   int
		nSamples  int
		nDim      int
		nCpu      int
		Name      string
	}{
		{
			Name:      "FullBatch",
			nFeatures: 8,
			nOutput:   3,
			nDim:      3,
			nCpu:      4,
			nSamples:  BatchSize * 2 * 4,
		},
		/*
			{
				Name:      "FullBatchOffset",
				nFeatures: 8,
				nOutput:   3,
				nDim:      3,
				nCpu:      4,
				nSamples:  BatchSize*2*4 + 9,
			},
			{
				Name:      "MoreThanCpuLessThanBatchOff",
				nFeatures: 8,
				nOutput:   3,
				nDim:      3,
				nCpu:      4,
				nSamples:  11,
			},
			{
				Name:      "MoreThanCpuLessThanBatchExact",
				nFeatures: 8,
				nOutput:   3,
				nDim:      3,
				nCpu:      4,
				nSamples:  16,
			},
		*/
	} {
		z := make([][]float64, test.nSamples)
		for i := range z {
			z[i] = make([]float64, test.nFeatures)
			for j := range z[i] {
				z[i][j] = rand.NormFloat64() * 10
			}
		}
		output := make([][]float64, test.nSamples)
		for i := range output {
			output[i] = make([]float64, test.nOutput)
		}

		featureWeights := common.NewSlicedSlice(test.nFeatures, test.nOutput)
		for i := range featureWeights.Slice {
			featureWeights.Slice[i] = rand.NormFloat64()*2.7 + 2
		}
		losser := loss.ManhattanDistance{}
		regularizer := &regularize.TwoNorm{Gamma: 0.001}

		// Find the loss and derivative by prediction
		var indLoss float64
		predOutput := make([]float64, test.nOutput)
		dLossDPredTmp := make([]float64, test.nOutput)
		dLossDWeightTmp := common.NewSlicedSlice(test.nFeatures, test.nOutput)
		dLossDWeightInd := common.NewSlicedSlice(test.nFeatures, test.nOutput)
		for i := range z {
			indLoss += PredLossDeriv(z[i], output[i], featureWeights, losser, regularizer, dLossDWeightTmp, predOutput, dLossDPredTmp)
			for j := range dLossDWeightInd.Slice {
				dLossDWeightInd.Slice[j] += dLossDWeightTmp.Slice[j]
			}
		}

		// Find the loss and deriv sequentially
		dLossDWeightSeq := common.NewSlicedSlice(test.nFeatures, test.nOutput)
		seqLoss := SeqLossDeriv(z, output, featureWeights, losser, regularizer, dLossDWeightSeq, dLossDWeightTmp, predOutput, dLossDPredTmp)

		if math.Abs(seqLoss-indLoss)/math.Abs(indLoss) > 1e-14 {
			t.Errorf("Sequential loss doesn't match ind loss for case %v. Expected %v, found %v", test.Name, indLoss, seqLoss)
		}
		if !floats.EqualApprox(dLossDWeightSeq.Slice, dLossDWeightInd.Slice, 1e-14) {
			t.Errorf("Sequential derivative doesn't match ind derivative for case %v", test.Name)
		}

		parLoss, dLossDWeightPar := ParLossDeriv(z, output, featureWeights, losser, regularizer)
		if math.Abs(indLoss-parLoss)/math.Abs(indLoss) > 1e-14 {
			fmt.Println(indLoss)
			fmt.Println(parLoss)
			fmt.Println(indLoss - parLoss)
			fmt.Println(math.Abs(indLoss-parLoss) / math.Abs(indLoss))
			t.Errorf("Parallel loss doesn't match ind loss for case %v. Expected %v, found %v", test.Name, indLoss, parLoss)
		}
		if !floats.EqualApprox(dLossDWeightPar.Slice, dLossDWeightSeq.Slice, 1e-14) {
			t.Errorf("Derivatives don't match for case %z", test.Name)
		}

	}
}
