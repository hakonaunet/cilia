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

void OscillatorView::renderCircles(sf::RenderWindow& window) {
    const auto& gridData = grid_.getGrid();
    float cellWidth = window.getSize().x / static_cast<float>(gridData[0].size());
    float cellHeight = window.getSize().y / static_cast<float>(gridData.size());

    // Determine the size of the circle based on the smaller dimension of the cell
    float radius = std::min(cellWidth, cellHeight) / 2.0f;

    sf::CircleShape circle(radius);
    circle.setOrigin(radius, radius); // Set the origin to the center of the circle for proper positioning

    for (size_t i = 0; i < gridData.size(); ++i) {
        for (size_t j = 0; j < gridData[i].size(); ++j) {
            circle.setFillColor(angleToColor(gridData[i][j].getAngle()));
            circle.setPosition(j * cellWidth + radius, i * cellHeight + radius); // Position the center of the circle
            window.draw(circle);
        }
    }
}