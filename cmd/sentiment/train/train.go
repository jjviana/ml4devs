package main

import (
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"strconv"

	"github.com/jjviana/ml4devs/pkg/ml"
)

func main() {

	fmt.Printf("CPU profiling enabled\n")
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	defer f.Close() // error handling omitted for example
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()

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

const numEpochs = 1000
const learningRate = 0.001
