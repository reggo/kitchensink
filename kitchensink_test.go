package kitchensink

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/gonum/blas/cblas"
	"github.com/gonum/matrix/mat64"
)

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
}