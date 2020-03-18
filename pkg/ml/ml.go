package ml

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"strconv"
)

//Model is the Machine Learning model we are trying to learn
//A linear model is of the form:
// y = c0*dfeature[0]+c1*feature[1]+...+cN*feature[n] + bias
type Model struct {
	Bias             float64
	Coeficients      []float64
	MinFeatureValues []float64
	MaxFeatureValues []float64
}

//Example is a single data point consisting of a feature set and a label
type Example struct {
	Features []float64
	Label    float64
}

//ReadCSVDataSet reads a CSV dataset
func ReadCSVDataSet(fileName string) ([]Example, error) {
	inputFile, err := os.Open(fileName)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %w", fileName, err)
	}
	reader := csv.NewReader(inputFile)
	reader.Comma = ';'
	var example Example
	//Ignore the first line (header)
	reader.Read()
	record, err := reader.Read()

	dataSet := make([]Example, 0)
	for err == nil {

		if len(record) < 10 {
			return nil, fmt.Errorf("error: expected 10 values, found %d", len(record))

		}
		example = Example{Features: make([]float64, len(record)-1)}

		for i := 0; i < len(example.Features); i++ {
			example.Features[i], err = strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, fmt.Errorf("Error parsing feature value (%s): %w ", record[i], err)

			}

		}
		example.Label, err = strconv.ParseFloat(record[len(record)-1], 64)

		if err != nil {
			return nil, fmt.Errorf("error parsing label (%s): %w", record[len(record)-1], err)
		}
		dataSet = append(dataSet, example)

		record, err = reader.Read()
	}

	if err != io.EOF {
		return nil, fmt.Errorf("Error: %s", err)

	}
	return dataSet, nil
}

//Train executes the training loop
func Train(dataSet []Example, learningRate float64, numEpochs int) (Model, error) {

	//Assumes the dataset has been normalized

	min, max, err := NormalizeDataSetFeatures(dataSet)

	if err != nil {
		return Model{}, fmt.Errorf("error normalizing dataset: %w", err)
	}
	model := Model{Coeficients: make([]float64, len(dataSet[0].Features)),
		MinFeatureValues: min, MaxFeatureValues: max}

	for epoch := 0; epoch < numEpochs; epoch++ {

		sumError := 0.0
		for i := 0; i < len(dataSet); i++ {
			prediction := Predict(model, dataSet[i])
			error := prediction - dataSet[i].Label
			sumError += error * error
			model.Bias -= learningRate * error

			for j := 0; j < len(model.Coeficients); j++ {
				model.Coeficients[j] -= learningRate * error * dataSet[i].Features[j]

			}

		}

		loss := math.Sqrt(sumError / float64(len(dataSet)))
		fmt.Printf("Epoch %d error %.3f\n", epoch, loss)

	}

	return model, nil

}

//NormalizeDataSetFeatures normalize the features in the dataset
//Returns two arays containing the minimum and the maximum value of each feature
//(for future use during inference)
func NormalizeDataSetFeatures(dataSet []Example) ([]float64, []float64, error) {

	if len(dataSet) < 1 {
		return nil, nil, fmt.Errorf("empty data set")
	}
	//Assumes all examples have the same number of features
	maxValues := make([]float64, len(dataSet[0].Features))
	minValues := make([]float64, len(dataSet[0].Features))

	//Initialize min and max
	for i := 0; i < len(maxValues); i++ {
		maxValues[i] = -math.MaxFloat64
		minValues[i] = math.MaxFloat64

	}

	//Find min and max
	for i := 0; i < len(dataSet); i++ {
		if len(dataSet[i].Features) < len(maxValues) {
			return nil, nil, fmt.Errorf("expected %d features, found %d ", len(maxValues), len(dataSet[i].Features))
		}
		for j := 0; j < len(maxValues); j++ {
			maxValues[j] = math.Max(maxValues[j], dataSet[i].Features[j])
			minValues[j] = math.Min(minValues[j], dataSet[i].Features[j])
		}
	}

	//Normalize by min and max

	NormalizeDatasetFeaturesWithLimits(dataSet, maxValues, minValues)

	return minValues, maxValues, nil

}

//NormalizeDatasetFeaturesWithLimits normalizes the data set features in place, with the provided minimum and maximum feature lengths
func NormalizeDatasetFeaturesWithLimits(dataSet []Example, maxValues []float64, minValues []float64) {
	for i := 0; i < len(dataSet); i++ {
		for j := 0; j < len(maxValues); j++ {
			dataSet[i].Features[j] = (dataSet[i].Features[j] - minValues[j]) / (maxValues[j] - minValues[j])
		}

	}
}

//Predict makes a prediction for a single example,
//given a model.
func Predict(model Model, example Example) float64 {

	result := model.Bias

	for i := 0; i < len(example.Features); i++ {
		result += model.Coeficients[i] * example.Features[i]
	}
	return result

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

	NormalizeDatasetFeaturesWithLimits(dataSet, model.MaxFeatureValues, model.MinFeatureValues)
	//Assuming loss is RMSE

	sumError := 0.0
	for _, example := range dataSet {
		prediction := Predict(model, example)
		error := prediction - example.Label
		sumError += error * error
		listener(example, prediction)

	}

	return math.Sqrt(sumError / float64(len(dataSet)))

}
