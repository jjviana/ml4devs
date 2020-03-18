package main

import (
	"fmt"
	"os"

	"github.com/jjviana/ml4devs/pkg/ml"
)

func main() {

	if len(os.Args) < 3 {
		fmt.Println("Usage: test <model file> <dataset>")
		return
	}

	model, err := ml.LoadModel(os.Args[1])
	if err != nil {
		fmt.Printf("Error loading model: %s\n", err)
		return
	}

	datasetFile := os.Args[2]
	dataSet, err := ml.ReadCSVDataSet(datasetFile)
	if err != nil {
		fmt.Printf("Error loading dataset: %s\n", err)
		return
	}

	acuracy := ml.Test(model, dataSet, func(example ml.Example, prediction float64) {
		fmt.Printf("Word: %s , Label %.03f, Prediction: %.03f\n", example.Word, example.Label, prediction)
	})

	fmt.Printf("\nAccuracy: %.03f%%\n", acuracy)

}
