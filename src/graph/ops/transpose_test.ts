/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { ENV } from '../../environment';
import { Array2D } from '../../math/ndarray';
import { expectArraysEqual } from '../../test_util';
import { Tensor } from '../graph';
import { SummedTensorArrayMap, TensorArrayMap } from '../tensor_array_map';
import { Transpose } from './transpose';

describe('transpose operation', () => {
  const math = ENV.math;

  let xTensor: Tensor;
  let yTensor: Tensor;
  let transposeOp: Transpose;
  let activations: TensorArrayMap;
  let gradients: SummedTensorArrayMap;

  beforeEach(() => {
    activations = new TensorArrayMap();
    gradients = new SummedTensorArrayMap(math);
  });

  afterEach(() => {
    activations.disposeArray(xTensor);
    activations.disposeArray(yTensor);
    gradients.disposeArray(xTensor);
    gradients.disposeArray(yTensor);
  });

  it('transpose', () => {
    const x = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    const dy = Array2D.new([3, 2], [[1, 4], [2, 5], [3, 6]]);

    xTensor = new Tensor(x.shape);
    yTensor = new Tensor(x.shape);

    activations.set(xTensor, x);

    transposeOp = new Transpose(xTensor, yTensor);
    transposeOp.feedForward(math, activations);
    const y = activations.get(yTensor);

    expect(y.shape).toEqual([3, 2]);
    expectArraysEqual(y, dy);

    gradients.add(yTensor, dy);

    transposeOp.backProp(math, activations, gradients);

    const dx = gradients.get(xTensor);

    expect(dx.shape).toEqual([2, 3]);
    expectArraysEqual(dx, x);
  });
});
