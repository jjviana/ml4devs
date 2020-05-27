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

	"github.com/nlpodyssey/spago/pkg/ml/nn/sparse_stochastic_linear"

	"github.com/nlpodyssey/spago/pkg/ml/nn"

	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/sgd"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

//Model is the Machine Learning model we are trying to learn
//A linear model is of the form:
// y = c0*dfeature[0]+c1*feature[1]+...+cN*feature[n] + bias
type Model struct {
	internalModel *sparse_stochastic_linear.Model
	Ngrams        int
}

//Example is a single data point consisting of a feature set and a label
type Example struct {
	Sentence string
	Features mat.Matrix
	Label    mat.Matrix
}

var HashTableSize = uint32(math.Pow(2, 21))

//FeatureHash takes a sentence and returns a representation of it based on the hashing trick.
//It uses the provided hash function and return a float64 slice of size HashTableSize
func FeatureHash(features []string, hash hash.Hash32, out mat.Matrix) {

	for _, feature := range features {
		hash.Reset()
		hash.Write([]byte(feature))
		index := hash.Sum32() % HashTableSize
		out.Set(int(index), 0, 1)
	}

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

	dataSet := make([]Example, 0)
	hashFunction := fnv.New32a()

	for err == nil {

		if len(record) < 2 {
			return nil, fmt.Errorf("error: expected 2 values, found %d", len(record))

		}
		example = Example{Sentence: record[0], Features: mat.NewSparse(int(HashTableSize), 1)}
		words := strings.Split(example.Sentence, " ")
		if ngrams > 0 {
			words = makeNgrams(words, ngrams)
		}
		FeatureHash(words, hashFunction, example.Features)

		var label float64
		label, err = strconv.ParseFloat(record[len(record)-1], 64)

		if err != nil {
			fmt.Printf("Warn: error parsing label (%s): %s (ignoring example)\n", record[len(record)-1], err)
			err = nil
		} else {
			example.Label = mat.NewScalar(label)
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

	model := Model{internalModel: sparse_stochastic_linear.New(int(HashTableSize), 1, 10, 1),
		Ngrams: ngrams}

	optimizer := gd.NewOptimizer(sgd.New(sgd.NewConfig(learningRate, 0.0, false)))

	nn.TrackParamsForOptimization(model.internalModel, optimizer)

	for epoch := 0; epoch < numEpochs; epoch++ {

		sumError := 0.0
		for i := 0; i < len(dataSet); i++ {
			example := &dataSet[i]
			g := ag.NewGraph()
			x := g.NewVariable(example.Features, false)
			y := model.internalModel.NewProc(g).Forward(x)[0]

			error := g.Abs(g.Sub(g.NewVariable(example.Label, false), y))
			g.Backward(error)
			optimizer.Optimize()
			sumError += error.ScalarValue() * error.ScalarValue()

		}

		loss := math.Sqrt(sumError / float64(len(dataSet)))
		fmt.Printf("Epoch %d error %.3f\n", epoch, loss)

	}

	return model, nil

}

//Predict makes a prediction for a single example,
//given a model.
/*func Predict(model Model, example *Example) float64 {

	result := model.Bias

	for i := 0; i < len(example.Features); i++ {
		result += model.Coeficients[example.Features[i]]
	}
	return result

}*/

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
		//prediction := Predict(model, &example)
		prediction := 0.0
		// Since this is a classification problem, we will consider the treshold 0.5
		// (even though our loss function does not yet interprets the model output as probability)

		/*classificationPrediction := 0.0
		if prediction > 0.5 {
			classificationPrediction = 1.0
		}
		/*
			if classificationPrediction == example.Label {
				correct = correct + 1
			}
		*/

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
