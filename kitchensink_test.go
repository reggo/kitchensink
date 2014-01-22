package kitchensink

import (
	"fmt"
	"math"
	//"math/rand"
	//"runtime"
	"testing"

	//"github.com/gonum/blas/cblas"
	//"github.com/gonum/matrix/mat64"

	"github.com/gonum/floats"

	"github.com/reggo/regtest"
	"github.com/reggo/train"
)

var (
	sink = &Sink{}
	_    = train.Trainable(sink)
)

type sinkIniter struct {
	nFeatures int
	kernel    Kernel
	inputDim  int
	outputDim int
	name      string
}

var testSinks []*Sink

var sinkIniters []*sinkIniter = []*sinkIniter{

	{
		nFeatures: 10,
		kernel:    IsoSqExp{},
		inputDim:  5,
		outputDim: 3,
		name:      "nFeatures > inputDim > outputDim",
	},
	{
		nFeatures: 13,
		kernel:    IsoSqExp{},
		inputDim:  3,
		outputDim: 5,
	},
	{
		nFeatures: 2,
		kernel:    IsoSqExp{},
		inputDim:  3,
		outputDim: 5,
	},
	{
		nFeatures: 2,
		kernel:    IsoSqExp{},
		inputDim:  6,
		outputDim: 4,
	},
	{
		nFeatures: 8,
		kernel:    IsoSqExp{},
		inputDim:  15,
		outputDim: 3,
	},
	{
		nFeatures: 8,
		kernel:    IsoSqExp{},
		inputDim:  4,
		outputDim: 12,
	},
}

func init() {
	// Set up all of the test sinks
	for _, initer := range sinkIniters {
		s := NewSink(initer.nFeatures, initer.kernel, initer.inputDim, initer.outputDim)
		testSinks = append(testSinks, s)
	}

}

func TestGetAndSetParameters(t *testing.T) {
	for i, test := range sinkIniters {
		s := testSinks[i]
		numParameters := s.NumParameters()
		trueNparameters := test.nFeatures * test.outputDim
		if numParameters != trueNparameters {
			t.Errorf("case %v: NumParameter mismatch. expected %v, found %v", test.name, trueNparameters, numParameters)
		}
		fmt.Println(test.name)
		regtest.TestGetAndSetParameters(t, s, test.name)
	}
}

func TestInputOutputDim(t *testing.T) {
	for i, test := range sinkIniters {
		s := testSinks[i]
		regtest.TestInputOutputDim(t, s, test.inputDim, test.outputDim, test.name)
	}
}

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
		z := computeZ(test.x, test.feature, test.b, math.Sqrt(2.0/test.nFeatures))
		if floats.EqualWithinAbsOrRel(z, test.z, 1e-14, 1e-14) {
			t.Errorf("z mismatch for case %v. %v expected, %v found", test.name, test.z, z)
		}
	}
}

/*

// TODO: Add real tests

func testfunc(x float64) float64 {
	return math.Sin(x/20) + x*x + 200
}

func generateRandomSamples(n, nDim int) (x, y *mat64.Dense) {
	x = mat64.NewDense(n, nDim, nil)
	y = mat64.NewDense(n, nDim, nil)

	for i := 0; i < n; i++ {
		for j := 0; j < nDim; j++ {
			x.Set(i, j, rand.NormFloat64())
			y.Set(i, j, testfunc(x.At(i, j)))
		}
	}
	return
}

func TestKitchenSink(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())
	// generate data
	mat64.Register(cblas.Blas{})
	nDim := 1
	nTrain := 1600
	xTrain, yTrain := generateRandomSamples(nTrain, nDim)

	nTest := 10000
	xTest, yTest := generateRandomSamples(nTest, nDim)

	nFeatures := 300

	// generate z
	sigmaSq := 0.01

	kernel := &IsoSqExp{LogScale: math.Log(sigmaSq)}

	// Train the struct
	sink := NewSink(nFeatures, kernel)
	sink.Train(xTrain, yTrain, nil, nil, nil)

	// Predict on trained values
	for i := 0; i < nTrain; i++ {
		_, _ = sink.Predict(xTrain.RowView(i), nil)
		//fmt.Println(yTrain.At(i, 0), pred[0], yTrain.At(i, 0)-pred[0])
	}
	fmt.Println()
	// Predict on new values
	pred, err := sink.PredictBatch(xTest, nil)
	if err != nil {
		t.Errorf(err.Error())
	}
	if nTest < 1000 {
		for i := 0; i < nTest; i++ {
			fmt.Println(pred.At(i, 0), yTest.At(i, 0), yTest.At(i, 0)-pred.At(i, 0))
		}
	}

	/*
		// TODO: Test weights
		weights := make([]float64, nTrain)
		for i := range weights {
			weights[i] = rand.Float64()
		}
		sink.Train(xTrain, yTrain, weights, nil, nil)
*/

/*
	for i := range testPred {
		fmt.Println(yTest[i][0], "\t", testPred[i][0], "\t", yTest[i][0]-testPred[i][0])
	}
*/
//}
