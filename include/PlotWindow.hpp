// PlotWindow.h
#pragma once

#include <string>
#include <filesystem>

#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

class PlotWindow {
public:
    PlotWindow(const std::string& filePath);
    ~PlotWindow();
    void showPlot();
    void update();
    bool isOpen() const;
    void display();

private:
    const std::string filePath;
    sf::RenderWindow window;
    sf::Texture plotTexture;
    sf::Sprite plotSprite;
};