// OscillatorView.hpp
#ifndef OSCILLATOR_VIEW_HPP
#define OSCILLATOR_VIEW_HPP

#include <SFML/Graphics/RectangleShape.hpp>

#include "Grid.hpp"
#include "Utils.hpp"

class OscillatorView {
public:
    OscillatorView(const Grid& grid, int size);
    void renderSquares(sf::RenderWindow& window);
    void renderCircles(sf::RenderWindow& window);

private:
    const Grid& grid_;
    int size_;
};

#endif