package main

import "base:runtime"
import "core:log"
import "core:slice"
import "core:encoding/json"

// My machine learning library is needed:
// https://github.com/Alkamist/odin_machine_learning
import ml "machine_learning"
import "machine_learning/mlp"

main_context: runtime.Context

model: Model

image_data: [MNIST_IMAGE_SIZE]f32

main :: proc() {
	logger := log.create_console_logger()
	defer log.destroy_console_logger(logger)
	context.logger = logger

	ml.init(1024 * 1024)

	model = make_model()

	unmarshal_err := json.unmarshal(#load("model.json"), &model)
	if unmarshal_err != nil {
		log.error("Failed to unmarshal model data")
		return
	}
}

@(export)
get_image_data_pointer :: proc "c" () -> rawptr {
	return &image_data[0]
}

@(export)
predict :: proc "c" () -> u8 {
	context = main_context

	ml.clear()

	logits        := forward(model, image_data[:])
	probabilities := ml.softmax(logits)

	return u8(slice.max_index(probabilities.data[:]))
}

MNIST_IMAGE_SIZE  :: 784
MNIST_CLASS_COUNT :: 10

Model :: struct {
	mlp: mlp.Mlp,
	opt: ml.Optimizer,
}

make_model :: proc(allocator := context.allocator) -> (model: Model) {
	model.mlp = mlp.make(MNIST_IMAGE_SIZE, 128, MNIST_CLASS_COUNT, allocator=allocator)
	return
}

destroy_model :: proc(model: Model) {
	mlp.destroy(model.mlp)
}

forward :: proc(model: Model, input: []f32) -> ml.Array {
	return mlp.forward(model.mlp, ml.array(input))
}