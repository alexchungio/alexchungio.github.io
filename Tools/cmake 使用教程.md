# cmake 使用教程



### 生成构建文件

```
cmake {-DCMAKE_VERBOSE_MAKEFILE} {-GXcode} ../
```

### 执行构建

```
cmake --build 
```

```
make -j 4
```

### 执行安装

```
make install
```

## 错误日志

### /bin/ld: ***: undefined reference to 'cudaMalloc'...

* 错误 cmake 文件

  ```
  target_link_libraries(${name} cudart)
  target_link_libraries(${name} nvinfer)
  target_link_libraries(${name} utils)  # 自定义库路径
  ```

* 解决：修改 target_link 顺序，前移自定义库路径 ， 考虑 g++ 编译的依赖顺序导致

  ```
  target_link_libraries(${name} utils)
  target_link_libraries(${name} cudart)
  target_link_libraries(${name} nvinfer)
  ```


## 参考链接

* <https://cmake.org/cmake/help/latest/manual/cmake.1.html>