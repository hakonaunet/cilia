//Utils.cpp
#include "Utils.hpp"

sf::Color angleToColor(double angle) {
    double hue = angle / (2 * M_PI); // Normalize to [0, 1]

    int i = int(hue * 6);
    double f = hue * 6 - i;
    double q = 1 - f;

    switch (i % 6) {
        case 0: return sf::Color(255, 255 * f, 0);
        case 1: return sf::Color(255 * q, 255, 0);
        case 2: return sf::Color(0, 255, 255 * f);
        case 3: return sf::Color(0, 255 * q, 255);
        case 4: return sf::Color(255 * f, 0, 255);
        case 5: return sf::Color(255, 0, 255 * q);
    }
    return sf::Color(255, 255, 255); // Default to white, but this shouldn't happen
}
