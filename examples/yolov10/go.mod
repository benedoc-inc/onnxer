module github.com/benedoc-inc/onnxer/examples/yolov8

go 1.25

replace github.com/benedoc-inc/onnxer => ../..

require (
	github.com/fogleman/gg v1.3.0
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
	github.com/benedoc-inc/onnxer v0.0.0-00010101000000-000000000000
)

require (
	github.com/ebitengine/purego v0.9.0 // indirect
	github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0 // indirect
	golang.org/x/image v0.32.0 // indirect
)
