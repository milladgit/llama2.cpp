#pragma once

#include <vector>
#include <numeric>
#include <initializer_list>
#include <cassert>
#include <iostream>

template <typename DataType>
class Tensor {
public:
    using ShapeT = unsigned long long;

    Tensor() : dataAllocatedByMe(false), data(nullptr), isMemoryMapped(false), fdMemoryMapped(false) {}

    Tensor(const std::initializer_list<ShapeT>& newShape) : shape(std::begin(newShape), std::end(newShape)) {
//        assert(newShape.size() >= 1);

        initializeShape(newShape);
        const ShapeT totalSize = std::accumulate(shape.begin(), shape.end(), (ShapeT)1, std::multiplies<ShapeT>());
        dataVec = std::make_shared<std::vector<DataType>>(totalSize, 0);
        data = (DataType*)dataVec->data();
        dataAllocatedByMe = true;
        isMemoryMapped = false;

        offset = std::vector<ShapeT>(newShape.size(), 0);
    }
    Tensor(const std::vector<ShapeT>& newShape) : shape(newShape) {
//        assert(newShape.size() >= 1);

        initializeShape(newShape);
        const ShapeT totalSize = std::accumulate(shape.begin(), shape.end(), (ShapeT)1, std::multiplies<ShapeT>());
        dataVec = std::make_shared<std::vector<DataType>>(totalSize, 0);
        data = (DataType*)dataVec->data();
        dataAllocatedByMe = true;
        isMemoryMapped = false;

        offset = std::vector<ShapeT>(newShape.size(), 0);
    }

    Tensor(DataType* ptr, const std::vector<ShapeT>& newShape) {
//        assert(newShape.size() >= 1);

        initializeShape(newShape);
        data = ptr;
        dataAllocatedByMe = false;
        isMemoryMapped = true;

        offset = std::vector<ShapeT>(newShape.size(), 0);
    }

    ~Tensor() {
        if(isMemoryMapped && data) {
        }
    }

    DataType& get(int i) {
        assert(shape.size() == 1);
        i += offset[0];
        return data[i];
    }
    DataType& get(int i, int j) {
        assert(shape.size() == 2);
        i += offset[0];
        j += offset[1];
        return data[i * strides[0] + j];
    }
    DataType& get(int i, int j, int k) {
        assert(shape.size() == 3);
        i += offset[0];
        j += offset[1];
        k += offset[2];
        return data[i * strides[0] + j * strides[1] + k];
    }
    const DataType& get(int i) const {
        assert(shape.size() == 1);
        i += offset[0];
        return data[i];
    }
    const DataType& get(int i, int j) const {
        assert(shape.size() == 2);
        i += offset[0];
        j += offset[1];
        return data[i * strides[0] + j];
    }
    const DataType& get(int i, int j, int k) const {
        assert(shape.size() == 3);
        i += offset[0];
        j += offset[1];
        k += offset[2];
        return data[i * strides[0] + j * strides[1] + k];
    }

    DataType& operator()(int i) {
        return get(i);
    }
    DataType& operator()(int i, int j) {
        return get(i, j);
    }
    DataType& operator()(int i, int j, int k) {
        return get(i, j, k);
    }
    const DataType& operator()(int i) const {
        return get(i);
    }
    const DataType& operator()(int i, int j) const {
        return get(i, j);
    }
    const DataType& operator()(int i, int j, int k) const {
        return get(i, j, k);
    }

    DataType* getData() {
//        assert(shape.size() > 0);

        return data;
    }

    template<typename T>
    Tensor<DataType> cropWithOffset(const std::vector<T>& offset) const {
        assert(std::is_integral_v<T>);
        assert(offset.size() == shape.size());

        auto t = *this;
        if constexpr (std::is_same_v<T, ShapeT>) {
            t.offset = offset;
        } else {
            std::transform(offset.begin(), offset.end(), this->offset.begin(), [](const T& a) {return (ShapeT)a;});
        }
        return t;
    }

    template<typename T>
    Tensor<DataType> cropWithOffset(const std::initializer_list<T>& offset) {
        assert(std::is_integral_v<T>);

        std::cout << "    MILLAD:  " << __FILE__ << ":" << __LINE__ << "   " << offset.size() << "   " << shape.size() << std::endl;

        assert(offset.size() == shape.size());

        auto t = *this;
        if constexpr (std::is_same_v<T, ShapeT>) {
            t.offset = offset;
        } else {
            std::transform(offset.begin(), offset.end(), this->offset.begin(), [](const T& a) {return (ShapeT)a;});
        }
        return t;
    }

    const std::vector<ShapeT>& getShape() const {
        return shape;
    }
    ShapeT getShape(int i) {
        return shape[i];
    }

private:
    std::vector<ShapeT> shape;
    std::vector<ShapeT> strides;
    std::vector<ShapeT> offset;
    bool dataAllocatedByMe;
    DataType* data;
    std::shared_ptr<std::vector<DataType>> dataVec;

    bool isMemoryMapped;
    int fdMemoryMapped;

    void initializeShape(const std::vector<ShapeT>& newShape) {
        strides = std::vector<ShapeT>(newShape.size());
        if(newShape.size() == 0) {
            return;
        }
        if(newShape.size() == 1) {
            strides[0] = 1;
            return;
        }
        strides[newShape.size() - 1] = 1;
        for(int i=(int)newShape.size()-2;i>=0;i--) {
            strides[i] = newShape[i+1] * strides[i+1];
        }
        shape = newShape;
    }
};

//template <typename DataType>
//class MemoryMappedTensor : public Tensor<DataType> {
//public:
//    MemoryMappedTensor(const char* filename) {
//
//    }
//};
