# g++ 编译命令

## gcc 与 g++ 的区别

GCC:GNU Compiler Collection，GUN 编译器集合；gcc(GNU C Compiler)；g++(G++ C++ Compiler)

gcc 和 g++ 不是编译器，而是一种驱动器。gcc 和 g++ 根据参数中要编译的文件的类型，调用对应的GUN编译器，其中 gcc 调用 c 编译器，g++ 调用 c++ 编译器。

### 主要区别

* 对于 *.c和*.cpp文件，gcc 分别当做 c 和 cpp 文件编译

*  对于 *.c和*.cpp文件，g++则统一当做 cpp 文件编译

* 使用g++编译文件时，**g++会自动链接标准库STL，而gcc不会自动链接STL**

* gcc在编译C文件时，可使用的预定义宏是比较少的

* gcc在编译cpp文件时/g++在编译c文件和cpp文件时（这时候gcc和g++调用的都是cpp文件的编译器），会加入一些额外的宏，这些宏如下

  ```c++
  #define __GXX_WEAK__ 1
  #define __cplusplus 1
  #define __DEPRECATED 1
  #define __GNUG__ 4
  #define __EXCEPTIONS 1
  #define __private_extern__ extern
  
  ```

* 用gcc编译c++文件时，为了能够使用STL，需要加参数 –lstdc++

## g++ 命令参数

* -c

  编译选项，编译源文件，但不执行链接，产生 .o 文件

  ```
  g++ -c {compile-options} file.cpp
  ```

* -o

  链接选项，指定输出文件的名称

  ```
  g++ -o {target-name} {link-options} file1.o file2.o ... other-libraries
  ```

* -g

  编译和链接选项，为了 gdb 调试使用

* -w

  编译选项，关闭所有警告

* -E

  预处理后停止，不运行编译器

* -S

  编译阶段结束后停止，不组装（assemble）, 输出 .s 文件

* -Wall

  产生警告(warning)信息

* -v

  打印执行命令

## 参考资料

* <https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Overall-Options.html#Overall-Options>
* <https://www.cs.bu.edu/fac/gkollios/cs113/Usingg++.html>
* <https://www.zhihu.com/question/20940822/answer/536826078>
