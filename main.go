package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func readData(path string) ([][]float32, []int) {
	// Open the file
	csvfile, err := os.Open(path)
	if err != nil {
		panic(err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	// Read all the records
	records, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	// Create slices to hold the data
	var data [][]float32
	var labels []int

	// Iterate through the records
	for _, record := range records[1:] {
		// Convert the label to int and add it to labels
		label, err := strconv.Atoi(record[0])
		if err != nil {
			panic(err)
		}
		labels = append(labels, label)

		var image []float32

		// Convert the rest of the values to float32 and add them to data
		for _, value := range record[1:] {
			// Convert string to int first
			intValue, err := strconv.Atoi(value)
			if err != nil {
				panic(err)
			}
			// Normalize the value to 0-1 range and convert to float32
			floatValue := float32(intValue) / 255.0
			image = append(image, floatValue)
		}

		data = append(data, image)
	}

	return data, labels
}

type Model struct {
	Weights []*Tensor
	Biases  []*Tensor
}

func NewModel(weights []*Tensor, biases []*Tensor) *Model {
	return &Model{
		Weights: weights,
		Biases:  biases,
	}
}

func (m *Model) Forward(x *Tensor) *Tensor {
	// Note to self: this loop is fine
	for i := 0; i < len(m.Weights); i++ {
		x = x.MatMul(m.Weights[i]).Add(m.Biases[i]).Sigmoid()
	}
	return x
}

func main() {
	model := NewModel(
		[]*Tensor{
			RandomTensor(28*28, 10),
		},
		[]*Tensor{
			RandomTensor(1, 10),
		},
	)

	images, labels := readData("./data/mnist_train.csv")

	const learningRate = 0.001

	for i := 0; i < len(images); i++ {
		output := model.Forward(NewTensor(images[i], []int{1, 28 * 28}, nil, NoOp))

		label := NewTensor([]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, []int{1, 10}, nil, NoOp)
		label.data[labels[i]] = 1

		diff := output.Sub(label)
		loss := diff.Mul(diff)

		if i%1000 == 0 {
			fmt.Println("Loss:", mean(loss.data))
		}

		// Zero the gradients
		for i := 0; i < len(model.Weights); i++ {
			model.Weights[i].grad = make([]float32, len(model.Weights[i].grad))
			model.Biases[i].grad = make([]float32, len(model.Biases[i].grad))
		}

		// Calculate the gradients
		loss.Backward([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})

		// Step the model
		for i := 0; i < len(model.Weights); i++ {
			model.Weights[i].data = sub(model.Weights[i].data, scale(model.Weights[i].grad, learningRate))
			model.Biases[i].data = sub(model.Biases[i].data, scale(model.Biases[i].grad, learningRate))
		}
	}

	images, labels = readData("./data/mnist_test.csv")

	var correct int

	for i := 0; i < len(images); i++ {
		output := model.Forward(NewTensor(images[i], []int{1, 28 * 28}, nil, NoOp))
		var predictedLabel int
		var maxOutput float32
		for i, val := range output.data {
			if val > maxOutput {
				maxOutput = val
				predictedLabel = i
			}
		}
		if labels[i] == predictedLabel {
			correct += 1
		}
	}

	accuracy := float32(correct) / float32(len(images)) * 100.0
	fmt.Printf("Test accuracy: %.2f%%\n", accuracy)
}

type OpType int

const (
	// Binary Ops
	NoOp OpType = iota
	AddOp
	SubOp
	MulOp
	MatMulOp

	// Unary Ops
	SigmoidOp
)

type Tensor struct {
	data    []float32
	grad    []float32
	shape   []int
	parents []*Tensor
	op      OpType
}

func NewTensor(data []float32, shape []int, parents []*Tensor, op OpType) *Tensor {
	return &Tensor{
		data:    data,
		grad:    make([]float32, len(data)),
		shape:   shape,
		parents: parents,
		op:      op,
	}
}

func size(shape []int) int {
	s := 1
	for i := 0; i < len(shape); i++ {
		s *= shape[i]
	}
	return s
}

func RandomTensor(shape ...int) *Tensor {
	data := make([]float32, size(shape))
	for i := 0; i < len(data); i++ {
		data[i] = rand.Float32()
	}
	return NewTensor(data, shape, nil, NoOp)
}

func check_shapes(a, b []int) {
	if len(a) != len(b) {
		panic("Ranks don't match")
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			panic("Dimensions don't match")
		}
	}
}

func add(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("Sizes don't match")
	}
	output := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = a[i] + b[i]
	}
	return output
}

func sub(a, b []float32) []float32 {
	return add(a, neg(b))
}

func scale(a []float32, s float32) []float32 {
	output := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = a[i] * s
	}
	return output
}

func neg(a []float32) []float32 {
	return scale(a, -1)
}

func mul(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("Sizes don't match")
	}
	output := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = a[i] * b[i]
	}
	return output
}

func sigmoid(a []float32) []float32 {
	output := make([]float32, len(a))
	for i := 0; i < len(output); i++ {
		output[i] = float32(1 / (1 + math.Exp(float64(-a[i]))))
	}
	return output
}

func mean(a []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i]
	}
	return sum / float32(len(a))
}

func (t1 *Tensor) Add(t2 *Tensor) *Tensor {
	check_shapes(t1.shape, t2.shape)
	return NewTensor(add(t1.data, t2.data), t1.shape, []*Tensor{t1, t2}, AddOp)
}

func (t1 *Tensor) Sub(t2 *Tensor) *Tensor {
	check_shapes(t1.shape, t2.shape)
	return NewTensor(sub(t1.data, t2.data), t1.shape, []*Tensor{t1, t2}, SubOp)
}

func (t1 *Tensor) Mul(t2 *Tensor) *Tensor {
	check_shapes(t1.shape, t2.shape)
	return NewTensor(mul(t1.data, t2.data), t1.shape, []*Tensor{t1, t2}, MulOp)
}

func (t1 *Tensor) Sigmoid() *Tensor {
	return NewTensor(sigmoid(t1.data), t1.shape, []*Tensor{t1}, SigmoidOp)
}

func (t1 *Tensor) MatMul(t2 *Tensor) *Tensor {
	if t1.shape[1] != t2.shape[0] {
		panic("Inner-dimensions don't match")
	}
	output := make([]float32, t1.shape[0]*t2.shape[1])
	for i := 0; i < t1.shape[0]; i++ {
		for j := 0; j < t2.shape[1]; j++ {
			for k := 0; k < t1.shape[1]; k++ {
				output[i*t2.shape[1]+j] += t1.data[i*t1.shape[1]+k] * t2.data[k*t2.shape[1]+j]
			}
		}
	}
	return NewTensor(output, []int{t1.shape[0], t2.shape[1]}, []*Tensor{t1, t2}, MatMulOp)
}

// TODO: something fishy is going on here
func (t *Tensor) Backward(grad []float32) {
	// apparently, the gradient should be added
	t.grad = add(t.grad, grad)

	if t.parents != nil {
		switch t.op {
		case AddOp:
			t.parents[0].Backward(t.grad)
			t.parents[1].Backward(t.grad)
		case SubOp:
			t.parents[0].Backward(t.grad)
			t.parents[1].Backward(neg(t.grad))
		case MulOp:
			t.parents[0].Backward(mul(t.grad, t.parents[1].data))
			t.parents[1].Backward(mul(t.grad, t.parents[0].data))
		case MatMulOp:
			t1 := t.parents[0]
			t2 := t.parents[1]

			t1Grad := make([]float32, len(t1.data))
			t2Grad := make([]float32, len(t2.data))

			// Calculate gradient with respect to t1
			for i := 0; i < t1.shape[0]; i++ {
				for k := 0; k < t1.shape[1]; k++ {
					for j := 0; j < t2.shape[1]; j++ {
						t1Grad[i*t1.shape[1]+k] += t.grad[i*t2.shape[1]+j] * t2.data[k*t2.shape[1]+j]
					}
				}
			}
			t1.Backward(t1Grad)

			// Calculate gradient with respect to t2
			for k := 0; k < t2.shape[0]; k++ {
				for j := 0; j < t2.shape[1]; j++ {
					for i := 0; i < t1.shape[0]; i++ {
						t2Grad[k*t2.shape[1]+j] += t1.data[i*t1.shape[1]+k] * t.grad[i*t2.shape[1]+j]
					}
				}
			}
			t2.Backward(t2Grad)
		case SigmoidOp:
			sigmoidGrad := make([]float32, len(t.data))
			for i := 0; i < len(t.data); i++ {
				sig := float32(1 / (1 + math.Exp(float64(-t.data[i]))))
				sigmoidGrad[i] = t.grad[i] * sig * (1 - sig)
			}
			t.parents[0].Backward(sigmoidGrad)
		}
	}
}
