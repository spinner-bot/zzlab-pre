#include <iostream>
#include <cmath>
using namespace std;

const double PI = 3.14159265;

// 1. 定义抽象基类 Shape
class Shape {
public:
    // 纯虚函数：计算面积
    virtual double area() = 0;
    // 纯虚函数：计算周长
    virtual double perimeter() = 0;
    // 虚析构函数（好习惯，防止内存泄漏）
    virtual ~Shape() {}
};

// 2. 圆形类
class Circle : public Shape {
private:
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() override { return PI * radius * radius; }
    double perimeter() override { return 2 * PI * radius; }
};

// 3. 矩形类
class Rectangle : public Shape {
private:
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() override { return width * height; }
    double perimeter() override { return 2 * (width + height); }
};

// 4. 三角形类 (这里简化为直角三角形或已知三边，为了演示方便使用海伦公式计算任意三角形)
class Triangle : public Shape {
private:
    double a, b, c; // 三边长
public:
    Triangle(double side_a, double side_b, double side_c) : a(side_a), b(side_b), c(side_c) {}

    double perimeter() override { return a + b + c; }

    double area() override {
        // 海伦公式
        double p = perimeter() / 2.0;
        return sqrt(p * (p - a) * (p - b) * (p - c));
    }
};

int main() {
    // 创建对象
    Circle c(5);
    Rectangle r(4, 6);
    Triangle t(3, 4, 5); // 直角三角形

    // 使用基类指针数组实现动态绑定 (核心考点)
    Shape* shapes[3] = {&c, &r, &t};

    cout << "--- 图形计算结果 ---" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "图形 " << i + 1 << ": "
             << "面积 = " << shapes[i]->area()
             << ", 周长 = " << shapes[i]->perimeter() << endl;
    }

    return 0;
}
