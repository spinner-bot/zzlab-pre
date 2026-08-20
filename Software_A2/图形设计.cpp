#include <iostream>
using namespace std;

class Rational {
private:
    int num; // 分子
    int den; // 分母

    // 求最大公约数 (辗转相除法)
    int gcd(int a, int b) {
        a = a > 0 ? a : -a; // 取绝对值
        b = b > 0 ? b : -b;
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 约分函数：保证分母为正，且为最简分数
    void simplify() {
        if (den == 0) {
            cout << "错误：分母不能为0" << endl;
            return;
        }
        // 1. 符号处理：保证分母永远为正
        if (den < 0) {
            num = -num;
            den = -den;
        }
        // 2. 约分
        int common = gcd(num, den);
        num /= common;
        den /= common;
    }

public:
    // 构造函数
    Rational(int n = 0, int d = 1) {
        num = n;
        den = d;
        simplify(); // 构造时直接约分
    }

    // 加法重载
    Rational operator+(const Rational& r) {
        return Rational(num * r.den + r.num * den, den * r.den);
    }

    // 减法重载
    Rational operator-(const Rational& r) {
        return Rational(num * r.den - r.num * den, den * r.den);
    }

    // 乘法重载
    Rational operator*(const Rational& r) {
        return Rational(num * r.num, den * r.den);
    }

    // 除法重载
    Rational operator/(const Rational& r) {
        return Rational(num * r.den, den * r.num);
    }

    // 输出重载 (友元函数)
    friend ostream& operator<<(ostream& os, const Rational& r) {
        if (r.den == 1)
            os << r.num;
        else
            os << r.num << "/" << r.den;
        return os;
    }
};

int main() {
    // 测试代码
    Rational a(1, 2); // 1/2
    Rational b(1, 3); // 1/3
    Rational c(2, -4); // -1/2 (自动约分)

    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    cout << "c = " << c << endl;

    cout << "a + b = " << a + b << endl; // 5/6
    cout << "a - b = " << a - b << endl; // 1/6
    cout << "a * b = " << a * b << endl; // 1/6
    cout << "a / b = " << a / b << endl; // 3/2

    return 0;
}
