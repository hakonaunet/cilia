// OseenView.cpp

#include "OseenView.hpp"

OseenView::OseenView(const Oseen& oseen) : oseen_(oseen) {}

void OseenView::prepare(const sf::RenderWindow& window) {
    // Get the dimensions of the window
    sf::Vector2u windowSize = window.getSize();

    // Calculate the scaling factor based on the grid spacing, width, and height
    double scale = std::min(windowSize.x / (oseen_.getGridSpacing() * oseen_.getWidth()),
                            windowSize.y / (oseen_.getGridSpacing() * oseen_.getHeight()));

    // Scale the cilia radius
    scaledRadius_ = oseen_.getCiliaRadius() * scale;

    const auto& positions = oseen_.getPositions();

    // Pre-calculate the scaled positions with padding
    scaledPositions_.resize(positions.rows());
    for (int i = 0; i < positions.rows(); ++i) {
        scaledPositions_[i].push_back(sf::Vector2f(
            positions(i, 0) * scale + scaledRadius_,
            positions(i, 1) * scale + scaledRadius_
        ));
    }
}

void OseenView::renderCircles(sf::RenderWindow& window) {
    sf::CircleShape circle(scaledRadius_);
    circle.setOrigin(scaledRadius_, scaledRadius_);

    const auto& angles = oseen_.getAngles();

    // Draw the circles
    for (size_t i = 0; i < scaledPositions_.size(); ++i) {
        for (size_t j = 0; j < scaledPositions_[i].size(); ++j) {
            circle.setPosition(scaledPositions_[i][j]);
            circle.setFillColor(angleToColor(angles(i)));
            window.draw(circle);
        }
    }
}

void OseenView::renderLines(sf::RenderWindow& window) {
    const auto& angles = oseen_.getAngles();

    // Draw the lines
    for (size_t i = 0; i < scaledPositions_.size(); ++i) {
        for (size_t j = 0; j < scaledPositions_[i].size(); ++j) {
            sf::RectangleShape line(sf::Vector2f(scaledRadius_, scaledRadius_ / 4));
            line.setOrigin(0, line.getSize().y / 2);
            line.setPosition(scaledPositions_[i][j]);
            line.setRotation(angles(i) * 180 / M_PI); // Convert angle from radians to degrees
            line.setFillColor(angleToColor(angles(i)));
            window.draw(line);
        }   
    }
}

void OseenView::renderSquares(sf::RenderWindow& window) {
    const auto& positions = oseen_.getPositions();
    double scale = std::min(window.getSize().x / (oseen_.getGridSpacing() * oseen_.getWidth()),
                            window.getSize().y / (oseen_.getGridSpacing() * oseen_.getHeight()));
    sf::RectangleShape rectangle(sf::Vector2f(scale, scale));
    const auto& angles = oseen_.getAngles();

    for (Eigen::Index i = 0; i < positions.rows(); ++i) {
        for (std::vector<sf::Vector2<float>>::size_type j = 0; j < scaledPositions_[i].size(); ++j) {
            rectangle.setFillColor(angleToColor(angles(i)));
            rectangle.setPosition(scaledPositions_[i][j]);
            window.draw(rectangle);
        }
    }
}