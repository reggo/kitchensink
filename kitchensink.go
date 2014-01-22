package kitchensink

import (
	"errors"
	"math"
	"math/rand"

	//"github.com/reggo/common"
	//"github.com/reggo/loss"
	predHelp "github.com/reggo/predict"
	//"github.com/reggo/regularize"
	"github.com/reggo/train"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// TODO: Generalize fitting method to allow the space binning thing

type Sink struct {
	kernel    Kernel
	nFeatures int

	inputDim       int
	outputDim      int
	features       *mat64.Dense // Index is feature number then
	featureWeights *mat64.Dense // Index is feature number then output
	b              []float64    // offsets from feature map
}

// NewSink returns a sink struct with the defaults
func NewSink(nFeatures int, kernel Kernel, inputDim, outputDim int) *Sink {
	sink := &Sink{
		nFeatures: nFeatures,
		kernel:    kernel,
		inputDim:  inputDim,
		outputDim: outputDim,
	}
	features := mat64.NewDense(sink.nFeatures, inputDim, nil)
	sink.kernel.Generate(sink.nFeatures, inputDim, features)
	sink.features = features
	sink.featureWeights = mat64.NewDense(sink.nFeatures, outputDim, nil)
	b := make([]float64, sink.nFeatures)
	for i := range b {
		b[i] = rand.Float64() * math.Pi * 2
	}
	sink.b = b
	return sink
}

func (s *Sink) InputDim() int {
	return s.inputDim
}

func (s *Sink) OutputDim() int {
	return s.outputDim
}

func (s *Sink) NumFeatures() int {
	return s.nFeatures
}

// IsLinear signifies that the prediction is a linear function of the parameters
func (s *Sink) IsLinear() bool {
	return true
}

// IsConvex signifies that the prediction is a convex funciton of the parameters
func (s *Sink) IsConvex() bool {
	return true
}

func (s *Sink) GrainSize() int {
	return 500
}

func (s *Sink) NumParameters() int {
	return s.outputDim * s.nFeatures
}

func (s *Sink) Parameters(p []float64) []float64 {
	if p == nil {
		p = make([]float64, s.NumParameters())
	} else {
		if len(p) != s.NumParameters() {
			panic("sink: parameter size mismatch")
		}
	}
	rm := s.featureWeights.RawMatrix()
	copy(p, rm.Data)
	return p
}

func (s *Sink) SetParameters(p []float64) {
	if len(p) != s.NumParameters() {
		panic("sink: parameter size mismatch")
	}
	rm := s.featureWeights.RawMatrix()
	copy(rm.Data, p)
}

// TODO: Add some comment about losser and regularizer are nil

/*
// Train trains the kitchen sink with the given inputs and outputs and the given weights
func (sink *Sink) Train(inputs, outputs mat64.Matrix, weights []float64, losser loss.Losser, regularizer regularize.Regularizer) (err error) {

	err = common.VerifyInputs(inputs, outputs, weights)
	if err != nil {
		return err
	}

	_, inputDim := inputs.Dims()
	_, outputDim := outputs.Dims()

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

	// See if outputs can be multiplied
	outputMul, outputIsMulMatrix := outputs.(train.MulMatrix)

	// Given the features, train the weights
	// See if can use simple linear solve
	if train.IsLinearSolveRegularizer(regularizer) && train.IsLinearSolveLosser(losser) && outputIsMulMatrix {
		// It can be solved with a linear solve, so construct wrapper and make the call
		lin := &linearSink{
			b:         b,
			nFeatures: sink.NumFeatures,
			features:  features,
		}
		var err error
		sink.featureWeights = train.LinearSolve(lin, nil, inputs, outputMul, nil, regularizer)
		if sink.featureWeights == nil {
			// TODO: Improve error message
			return errors.New("error training")
		}
		return err
	}
	panic("Not yet coded for non-SquaredDistance losser")
}
*/

// Predict returns the output at a given input. Returns nil if the length of the inputs
// does not match the trained number of inputs. The input value is unchanged, but
// will be modified during a call to the method
func (sink *Sink) Predict(input []float64, output []float64) ([]float64, error) {
	if len(input) != sink.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if len(output) != sink.outputDim {
		return nil, errors.New("output dimension mismatch")
	}

	if output == nil {
		output = make([]float64, sink.outputDim)
	}

	predict(input, sink.features, sink.b, sink.featureWeights, output)
	return output, nil
}

func (sink *Sink) PredictBatch(inputs mat64.Matrix, outputs mat64.Mutable) (mat64.Mutable, error) {
	batch := batchPredictor{
		features:       sink.features,
		featureWeights: sink.featureWeights,
		b:              sink.b,
	}
	return predHelp.BatchPredict(batch, inputs, outputs, sink.inputDim, sink.outputDim, sink.GrainSize())
}

// batchPredictor is a wrapper for BatchPredict to allow parallel predictions
type batchPredictor struct {
	features       *mat64.Dense
	featureWeights *mat64.Dense
	b              []float64
}

// There is no temporary memory involved, so can just return itself
func (b batchPredictor) NewPredictor() predHelp.Predictor {
	return b
}

func (b batchPredictor) Predict(input, output []float64) {
	predict(input, b.features, b.b, b.featureWeights, output)
}

// ComputeZ computes the value of z with the given feature vector and b value.
// Sqrt2OverD = math.Sqrt(2.0 / len(nFeatures))
func computeZ(input, feature []float64, b float64, sqrt2OverD float64) float64 {
	dot := floats.Dot(input, feature)
	return sqrt2OverD * (math.Cos(dot + b))
}

// wrapper for predict, assumes all inputs are correct
func predict(input []float64, features *mat64.Dense, b []float64, featureWeights *mat64.Dense, output []float64) {
	for i := range output {
		output[i] = 0
	}

	nFeatures, outputDim := features.Dims()

	sqrt2OverD := math.Sqrt(2.0 / float64(nFeatures))
	//for i, feature := range features {
	for i := 0; i < nFeatures; i++ {
		z := computeZ(input, features.RowView(i), b[i], sqrt2OverD)
		for j := 0; j < outputDim; j++ {
			output[j] += z * featureWeights.At(i, j)
		}
	}
}

// NewFeaturizer returns a featurizer for use in training routines.
func (s *Sink) NewFeaturizer() train.Featurizer {
	// The sink featurize method can be called in parallel normally, so
	// nothing is created
	return s
}

// Featurize computes the feature values for the input and stores them in
// place into Featurize
func (sink *Sink) Featurize(input, feature []float64) {
	sqrt2OverD := math.Sqrt(2.0 / float64(sink.nFeatures))
	for i := range feature {
		feature[i] = computeZ(input, sink.features.RowView(i), sink.b[i], sqrt2OverD)
	}
}

func (s *Sink) NewLossDeriver() train.LossDeriver {
	return &lossDerivWrapper{
		s: s,
	}
}

// TODO: Figure out how to couple this with the struct itself better to
// allow this to be exposed to other functions
// TODO: Should be something about precomputing or not precomputing all of the features
// Decouple settings from training to allow different things.
// Maybe don't have the Train at all for sinks? Just provide the train routines? Or Train on the type
// is simple, but reggo/train contains more complicated functions if necessary?
// DerivSink is a wrapper for training with gradient-based optimization
type lossDerivWrapper struct {
	s *Sink
}

func (d *lossDerivWrapper) Predict(input, predOutput []float64) {
	predict(input, d.s.features, d.s.b, d.s.featureWeights, predOutput)
}

func (d *lossDerivWrapper) Deriv(featurizedInput, predOutput, dLossDPred, dLossDWeight []float64) {
	// Form a matrix that has the underlying elements as dLossDWeight so the values are modified in place
	//lossMat := mat64.NewDense(d.s.nFeatures, d.s.outputDim, dLossDWeight)
	deriv(featurizedInput, dLossDPred, dLossDWeight)
}

func deriv(z []float64, dLossDPred []float64, dLossDWeight []float64) {
	// TODO: Can probably make this faster if we don't bother having the at, and instead work on the slice directly

	// dLossDWeight_ij = \sum_k dLoss/dPred_k * dPred_k / dWeight_j

	// Remember, the parameters are stored in row-major order

	nOutput := len(dLossDPred)
	// The prediction is just weights * z, so dPred_jDWeight_i = z_i
	// dLossDWeight = dLossDPred * dPredDWeight
	for i, zVal := range z {
		for j := 0; j < nOutput; j++ {
			dLossDWeight[i*nOutput+j] = zVal * dLossDPred[j]
		}
	}
}
