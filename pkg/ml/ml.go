package ml

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"hash"
	"hash/fnv"
	"io"
	"io/ioutil"
	"math"
	"os"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

//Model is the Machine Learning model we are trying to learn
//A linear model is of the form:
// y = c0*dfeature[0]+c1*feature[1]+...+cN*feature[n] + bias
type Model struct {
	Bias        float64
	Coeficients []float64
	Ngrams      int
}

//Example is a single data point consisting of a feature set and a label
type Example struct {
	Sentence string
	Features []uint32
	Label    float64
}

var HashTableSize = uint32(math.Pow(2, 21))

//FeatureHash takes a sentence and returns a representation of it based on the hashing trick.
//It uses the provided hash function and return a float64 slice of size HashTableSize
func FeatureHash(features []string, hash hash.Hash32) ([]uint32, error) {

	result := make([]uint32, 0)

	for _, feature := range features {
		hash.Reset()
		hash.Write([]byte(feature))
		index := hash.Sum32() % HashTableSize
		result = append(result, index)
	}
	return result, nil
}

func makeNgrams(words []string, ngrams int) []string {
	result := make([]string, 0)

	for i := 0; i < len(words); i++ {
		feature := words[i]
		result = append(result, feature)
		for j := 1; j < ngrams; j++ {
			if (i + j) < len(words) {
				feature = feature + "_" + words[i+j]
				result = append(result, feature)
			}

		}
	}
	return result

}

//ReadCSVDataSet reads a CSV dataset
func ReadCSVDataSet(fileName string, ngrams int) ([]Example, error) {
	inputFile, err := os.Open(fileName)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %w", fileName, err)
	}
	reader := csv.NewReader(inputFile)
	reader.Comma = ','
	var example Example

	//Ignore first line (header)

	record, err := reader.Read()

	record, err = reader.Read()

	dataSet := make([]Example, 0)
	hashFunction := fnv.New32a()

	for err == nil {

		if len(record) < 2 {
			return nil, fmt.Errorf("error: expected 2 values, found %d", len(record))

		}
		example = Example{Sentence: record[0]}
		words := strings.Split(example.Sentence, " ")
		if ngrams > 0 {
			words = makeNgrams(words, ngrams)
		}
		example.Features, err = FeatureHash(words, hashFunction)
		if err != nil {
			return nil, err
		}

		example.Label, err = strconv.ParseFloat(record[len(record)-1], 64)

		if err != nil {
			fmt.Printf("Warn: error parsing label (%s): %s (ignoring example)\n", record[len(record)-1], err)
			err = nil
		} else {
			dataSet = append(dataSet, example)
		}

		record, err = reader.Read()
	}

	if err != io.EOF {
		return nil, fmt.Errorf("Error: %s", err)

	}
	return dataSet, nil
}

//Train executes the training loop
func Train(dataSet []Example, learningRate float64, numEpochs int, ngrams int) (Model, error) {

	//Assumes the dataset has been normalized

	model := Model{Coeficients: make([]float64, HashTableSize),
		Ngrams: ngrams}

	for epoch := 0; epoch < numEpochs; epoch++ {

		sumError := 0.0
		for i := 0; i < len(dataSet); i++ {
			example := &dataSet[i]
			prediction := Predict(model, example)
			error := prediction - dataSet[i].Label
			errorWRTx := prediction * (1 - prediction)
			sumError += error
			gradient := learningRate * error * errorWRTx
			model.Bias -= gradient

			for j := 0; j < len(example.Features); j++ {
				model.Coeficients[example.Features[j]] -= gradient
			}

		}

		loss := sumError / float64(len(dataSet))
		fmt.Printf("Epoch %d error %.6f\n", epoch, loss)

	}

	return model, nil

}

//Predict makes a prediction for a single example,
//given a model.
func Predict(model Model, example *Example) float64 {

	result := model.Bias

	for i := 0; i < len(example.Features); i++ {
		result += model.Coeficients[example.Features[i]]
	}

	return sigmoid(result)

}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

//SaveModel saves a model to a file in JSON format.
func SaveModel(model Model, fileName string) error {

	content, err := json.MarshalIndent(model, " ", " ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(fileName, content, 0644)

}

//LoadModel loads a model from a file
func LoadModel(fileName string) (Model, error) {

	model := Model{}
	content, err := ioutil.ReadFile(fileName)
	if err != nil {
		return model, err
	}

	err = json.Unmarshal(content, &model)

	return model, err

}

type testListener func(Example, float64)

//Test tests the provided model in the provided dataset, returning the loss
func Test(model Model, dataSet []Example, listener testListener) float64 {

	correct := 0.0
	for _, example := range dataSet {
		prediction := Predict(model, &example)
		// Since this is a classification problem, we will consider the treshold 0.5
		// (even though our loss function does not yet interprets the model output as probability)

		classificationPrediction := 0.0
		if prediction > 0.5 {
			classificationPrediction = 1.0
		}
		if classificationPrediction == example.Label {
			correct = correct + 1
		}
		listener(example, prediction)

	}

	return correct / float64(len(dataSet))

}
func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}

func normalizeWord(word string) (string, error) {
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	result, _, err := transform.String(t, word)

	if err != nil {
		return "", fmt.Errorf("error normalizing word %s: $w", word, err)
	}
	return result, nil
}
