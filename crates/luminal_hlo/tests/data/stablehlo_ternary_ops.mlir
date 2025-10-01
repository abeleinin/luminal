module {
  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>) -> tensor<5xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<5xi1>, tensor<5xf32>    
    return %0 : tensor<5xf32>
  }
}
