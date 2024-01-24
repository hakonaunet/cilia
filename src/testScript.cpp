#include <iostream>
#include <cmath>

#define N 2
#define F0 1 
#define DT 0.001
#define PI 3.14159265358979323846

int main() {
    double q[N], w[N];
    int steps = 10000;
    double r12, v1x, v1y, v2x, v2y, f1x, f1y, f2x, f2y;

    q[0] = 0;
    q[1] = 0.1;

    w[0] = 1;
    w[1] = 1.1;


    for (int i = 0; i < steps; i++) {
        w[0] = F0;
        r12 = (5+cos(q[0])-cos(q[1]))*(5+cos(q[0])-cos(q[1]))+(sin(q[0])-sin(q[1]))*(sin(q[0])-sin(q[1]));
        r12 = sqrt(r12);
        f2x = -F0*(sin(q[1]));
        f2y = F0*(cos(q[1]));
        v1x = (1/r12)*f2x;
        v1x += 1/(r12*r12*r12)*((5+cos(q[0])-cos(q[1]))*f2x+(sin(q[0])-sin(q[1]))*f2y)*(5+cos(q[0])-cos(q[1]));
        v1y = (1/r12)*f2y;
        v1y += 1/(r12*r12*r12)*((5+cos(q[0])-cos(q[1]))*f2x+(sin(q[0])-sin(q[1]))*f2y)*(sin(q[0])-sin(q[1]));

        f1x = -F0*(sin(q[0]));
        f1y = F0*(cos(q[0]));
        v2x = (1/r12)*f1x;
        v2x += 1/(r12*r12*r12)*((5+cos(q[0])-cos(q[1]))*f1x+(sin(q[0])-sin(q[1]))*f1y)*(5+cos(q[0])-cos(q[1]));
        v2y = (1/r12)*f1y;
        v2y += 1/(r12*r12*r12)*((5+cos(q[0])-cos(q[1]))*f1x+(sin(q[0])-sin(q[1]))*f1y)*(sin(q[0])-sin(q[1]));

        w[0] += (-sin(q[0])*v1x+cos(q[0])*v1y);
        w[1] += (-sin(q[1])*v2x+cos(q[1])*v2y);

        // Euler method
        q[0] += w[0]*DT;
        q[1] += w[1]*DT;

        if (i % 1 == 0) {
            std::cout << "q[0]: " << cos(q[0]) << " q[1]: " << cos(q[1]) << std::endl;
        }
    }
    return 0;
}

