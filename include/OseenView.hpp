// OseenView.hpp

#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "Oseen.hpp"
#include "Utils.hpp"

class OseenView {
public:
    OseenView(const Oseen& oseen);

    void prepare(const sf::RenderWindow& window);
    void renderCircles(sf::RenderWindow& window);
    void renderLines(sf::RenderWindow& window);

private:
    const Oseen& oseen_;
    std::vector<std::vector<sf::Vector2f>> scaledPositions_;
    double scaledRadius_;
};