package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/jjviana/ml4devs/pkg/ml"
)

func main() {

	if len(os.Args) < 3 {
		fmt.Printf("Usage: sentiment <training file> <output file> [-ngrams N] \n")
		return
	}
	trainingFileName := os.Args[1]
	ngrams := 0
	if len(os.Args) == 5 && os.Args[3] == "-ngrams" {
		n, err := strconv.ParseInt(os.Args[4], 10, 32)
		if err != nil {
			fmt.Printf("Erro parsing ngrams: %s \n ", err)
			return
		}
		ngrams = int(n)
	}

	dataSet, err := ml.ReadCSVDataSet(trainingFileName, ngrams)
	if err != nil {
		fmt.Printf("Error reading dataset: %s \n", err)
		return
	}

	fmt.Printf("Read %d training examples\n", len(dataSet))

	model, err := ml.Train(dataSet, learningRate, numEpochs, ngrams)

	if err != nil {
		fmt.Printf("Error in training: %s ", err)
		return
	}
	err = ml.SaveModel(model, os.Args[2])
	if err != nil {
		fmt.Printf("Error saving model: %s \n", err)
	}

}

func printDataSet(dataSet []ml.Example) {

	for i := 0; i < len(dataSet); i++ {
		for j := 0; j < len(dataSet[i].Features); j++ {
			fmt.Printf("%.03f ", dataSet[i].Features[j])

		}
		fmt.Printf("->%.04f\n", dataSet[i].Label)
	}
}

const numEpochs = 1000
const learningRate = 0.001
