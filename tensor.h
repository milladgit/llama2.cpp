#pragma once

#include <vector>
#include <numeric>
#include <initializer_list>
#include <cassert>

template <typename DataType>
class Tensor {
public:
    using ShapeT = unsigned long long;

    Tensor() {}

    Tensor(const std::initializer_list<ShapeT>& newShape) : shape(std::begin(newShape), std::end(newShape)) {
        assert(newShape.size() >= 1);
        initializeShape(newShape);
        const ShapeT totalSize = std::accumulate(shape.begin(), shape.end(), (ShapeT)1, std::multiplies<ShapeT>());
        data = calloc(totalSize, sizeof(DataType));
        dataAllocatedByMe = true;
        isMemoryMapped = false;
    }
    Tensor(const std::vector<ShapeT>& newShape) : shape(newShape) {
        assert(newShape.size() >= 1);
        initializeShape(newShape);
        const ShapeT totalSize = std::accumulate(shape.begin(), shape.end(), (ShapeT)1, std::multiplies<ShapeT>());
        data = calloc(totalSize, sizeof(DataType));
        dataAllocatedByMe = true;
        isMemoryMapped = false;
    }

    Tensor(DataType* ptr, const std::vector<ShapeT>& newShape) {
        assert(newShape.size() >= 1);
        initializeShape(newShape);
        data = ptr;
        dataAllocatedByMe = false;
        isMemoryMapped = false;
    }

    ~Tensor() {
        if(dataAllocatedByMe) {
            free(data);
        }
    }

    DataType& get(int i) {
        assert(shape.size() == 1);
        return data[i];
    }

    DataType& get(int i, int j) {
        assert(shape.size() == 2);
        return data[i * strides[1] + j];
    }

    DataType& get(int i, int j, int k) {
        assert(shape.size() == 3);
        return data[i * strides[2] + j * strides[1] + k];
    }

    DataType* getData() {
        assert(shape.size() > 0);
        return data;
    }

private:
    std::vector<ShapeT> shape;
    std::vector<ShapeT> strides;
    bool dataAllocatedByMe;
    DataType* data;

    bool isMemoryMapped;
    int fdMemoryMapped;

    void initializeShape(const std::vector<int>& newShape) {
        strides = std::vector<int>(newShape.size());
        strides[newShape.size() - 1] = 1;
        for(int i=(int)newShape.size()-2;i>=0;i--) {
            strides[i] = newShape[i] * strides[i-1];
        }
    }
};
