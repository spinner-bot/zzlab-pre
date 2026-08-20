#include <iostream>
#include <stdexcept>
using namespace std;

class Rational {
private:
    int num; // 分子
    int den; // 分母

    // 求最大公约数 (辗转相除法)
    static int gcd(int a, int b) {
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
    // 构造函数重载
    Rational() : num(0), den(1) {}              // 默认构造：0
    Rational(int n) : num(n), den(1) {}         // 整数构造：n
    Rational(int n, int d) : num(n), den(d) {   // 分数构造：n/d
        if (d == 0) throw invalid_argument("分母不能为0");
        simplify();
    }

    // 四则运算（const 成员函数，不修改对象）
    Rational operator+(const Rational& r) const {
        return Rational(num * r.den + r.num * den, den * r.den);
    }
    Rational operator-(const Rational& r) const {
        return Rational(num * r.den - r.num * den, den * r.den);
    }
    Rational operator*(const Rational& r) const {
        return Rational(num * r.num, den * r.den);
    }
    Rational operator/(const Rational& r) const {
        if (r.num == 0) throw invalid_argument("除数不能为0");
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

    // 构造函数重载演示
    Rational d;      // 0
    Rational e(5);   // 5
    cout << "d = " << d << ", e = " << e << endl;

    // 数据安全性演示：分母为0 / 除以0 会抛异常
    try {
        Rational f(1, 0);
    } catch (const invalid_argument& ex) {
        cout << "捕获异常: " << ex.what() << endl;
    }
    try {
        Rational zero(0, 1);
        cout << "a / zero = " << a / zero << endl;
    } catch (const invalid_argument& ex) {
        cout << "捕获异常: " << ex.what() << endl;
    }

    return 0;
}
