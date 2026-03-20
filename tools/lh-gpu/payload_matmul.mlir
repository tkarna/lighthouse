module {
  func.func @payload(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %3 = linalg.matmul ins(%1, %2 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg0 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
}
