// OscillatorView.cpp
#include "OscillatorView.hpp"

OscillatorView::OscillatorView(const Grid& grid, int size)
    : grid_(grid), size_(size) {}

void OscillatorView::renderSquares(sf::RenderWindow& window) {
    const auto& gridData = grid_.getGrid();
    float cellWidth = window.getSize().x / static_cast<float>(gridData[0].size());
    float cellHeight = window.getSize().y / static_cast<float>(gridData.size());
    sf::RectangleShape rectangle(sf::Vector2f(cellWidth, cellHeight));
    for (size_t i = 0; i < gridData.size(); ++i) {
        for (size_t j = 0; j < gridData[i].size(); ++j) {
            rectangle.setFillColor(angleToColor(gridData[i][j].getAngle()));
            rectangle.setPosition(j * cellWidth, i * cellHeight);
            window.draw(rectangle);
        }
    }
}