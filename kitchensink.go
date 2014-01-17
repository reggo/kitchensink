package kitchensink

import (
	"math"
	"math/rand"

	"github.com/reggo/common"
	"github.com/reggo/kitchensink/util"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"
	"github.com/reggo/scale"
	"github.com/reggo/train"

	"github.com/gonum/matrix/mat64"
)

type Sink struct {
	Kernel       Kernel
	InputScaler  scale.Scaler
	OutputScaler scale.Scaler
	Loss         loss.Losser
	NumFeatures  int
	Regularizer  regularize.Regularizer

	inputDim       int
	outputDim      int
	features       *mat64.Dense // Index is feature number then
	featureWeights *mat64.Dense // Index is feature number then output
	b              []float64    // offsets from feature map
}

// NewSink returns a sink struct with the defaults
func NewSink(nFeatures int, kernel Kernel) *Sink {
	sink := &Sink{
		InputScaler:  &scale.Normal{},
		OutputScaler: &scale.Normal{},
		Loss:         loss.SquaredDistance{},
		NumFeatures:  nFeatures,
		Kernel:       kernel,
		Regularizer:  regularize.None{},
	}
	return sink
}

// TODO: Generalize fitting method to allow the space binning thing

// Train trains the kitchen sink with the given inputs and outputs and the given weights
// Currently only works for one output
func (sink *Sink) Train(inputs, outputs *mat64.Dense, weights []float64) (err error) {

	err = common.VerifyInputs(inputs, outputs, weights)
	if err != nil {
		return err
	}

	nSamples, inputDim := inputs.Dims()
	_, outputDim := outputs.Dims()

	if len(weights) == 0 {
		weights = make([]float64, nSamples)
		for i := range weights {
			weights[i] = 1
		}
	}

	scale.ScaleTrainingData(inputs, outputs, sink.InputScaler, sink.OutputScaler)
	defer func() {
		err = scale.UnscaleTrainingData(inputs, outputs, sink.InputScaler, sink.OutputScaler)
	}()

	// Generate the features
	features := mat64.NewDense(sink.NumFeatures, inputDim, nil)
	sink.Kernel.Generate(sink.NumFeatures, inputDim, features)
	sink.features = features
	sink.inputDim = inputDim
	sink.outputDim = outputDim

	b := make([]float64, sink.NumFeatures)
	for i := range b {
		b[i] = rand.Float64() * math.Pi * 2
	}
	sink.b = b

	// Given the features, train the weights
	// See if can use simple linear solve
	if train.IsLinearSolveRegularizer(sink.Regularizer) && train.IsLinearSolveLosser(sink.Loss) {
		// It can be solved with a linear solve, so construct wrapper and make the call
		lin := &linearSink{
			b:         b,
			nFeatures: sink.NumFeatures,
			features:  features,
		}
		var err error
		sink.featureWeights, err = train.LinearSolve(lin, inputs, outputs, nil, sink.Regularizer)
		return err
	}
	panic("Not yet coded for non-SquaredDistance losser")
}

/*
func trainSqDist(inputs, outputs *mat64.Dense, weights []float64, features [][]float64, b []float64) (featureWeights [][]float64) {



	nSamples := len(inputs)
	nFeatures := len(features)
	nOutputs := len(outputs[0])
	// Compute Z for all of the features, and store it in the matrix
	zMat := matrix.NewDense(nSamples, nFeatures, nil)
	sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			zMat.Set(i, j, util.ComputeZ(inputs[i], features[j], b[j], sqrt2OverD))
		}
	}

	yMat := matrix.NewDense(nSamples, nOutputs, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nOutputs; j++ {
			yMat.Set(i, j, outputs[i][j])
		}
	}

	alpha := matrix.Solve(zMat, yMat)

	featureWeights = make([][]float64, nFeatures)
	for i := range features {
		featureWeights[i] = make([]float64, nOutputs)
		for j := range featureWeights[i] {
			featureWeights[i][j] = alpha.At(i, j)
		}
	}
	return
}
*/

// Predict returns the output at a given input. Returns nil if the length of the inputs
// does not match the trained number of inputs. The input value is unchanged, but
// will be modified during a call to the method
func (sink *Sink) Predict(input []float64, output []float64) error {
	if len(input) != sink.inputDim {
		return nil
	}
	output = make([]float64, sink.outputDim)

	sink.InputScaler.Scale(input)
	defer sink.InputScaler.Unscale(input)

	util.Predict(input, sink.features, sink.b, sink.featureWeights, output)
	sink.OutputScaler.Unscale(output)
	return output
}

/*
// PredictSlice has the same behavior as Predict except it predicts a list of inputs concurrently
// It uses runtime.GOMAXPROCS to determine the level of concurrency
func (sink *Sink) PredictSlice(inputs *mat64.Dense) (outputs *mat64.Dense) {

	nSamples, inputDim := inputs.Dims()
	if len(input) != sink.inputDim {
		return nil
	}
	outputs = make([][]float64, len(inputs))
	for i := range outputs {
		outputs[i] = make([]float64, sink.outputDim)
	}

	scale.ScaleData(sink.InputScaler, inputs)
	defer scale.UnscaleData(sink.InputScaler, inputs)
	util.ParallelPredict(inputs, sink.features, sink.b, sink.featureWeights, outputs)
	scale.UnscaleData(sink.OutputScaler, outputs)
	return outputs
}
*/

// linearSink is an interface for having the kitchen sink algorithm
// be able to use the linear solve
type linearSink struct {
	features  *mat64.Dense
	b         []float64
	nFeatures int
}

func (l *linearSink) CanParallelize() bool {
	return true
}

func (l *linearSink) NumFeatures() int {
	return l.nFeatures
}

func (l *linearSink) Featurize(input, feature []float64) {
	// compute the
	sqrt2OverD := math.Sqrt(2.0 / float64(l.nFeatures))
	for i := range feature {
		feature[i] = util.ComputeZ(input, l.features.RowView(i), l.b[i], sqrt2OverD)
	}
}
