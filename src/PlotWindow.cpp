// PlotWindow.cpp
#include "PlotWindow.hpp"

PlotWindow::PlotWindow(const std::string& filePath) : window(sf::VideoMode(800, 600), "Plot Window") {
    if (!ImGui::SFML::Init(window)) {
        throw std::runtime_error("Failed to initialize ImGui-SFML");
    }
    std::string fullPath = "output_files/" + filePath;
    if (!plotTexture.loadFromFile(fullPath)) {
        throw std::runtime_error("Failed to load texture from file: " + fullPath);
    }
    plotSprite.setTexture(plotTexture);
}

PlotWindow::~PlotWindow() {
    ImGui::SFML::Shutdown();
}

void PlotWindow::showPlot() {
    sf::Event event;
    while (window.pollEvent(event)) {
        ImGui::SFML::ProcessEvent(window, event);

        if (event.type == sf::Event::Closed) {
            window.close();
        }
    }

    ImGui::SFML::Update(window, sf::seconds(1.f/60.f));

    if (ImGui::Begin("Plot")) {
        ImGui::Image(plotSprite);

        if (ImGui::Button("Save plot")) {
            std::string output_dir = "output_files";
            if (!std::filesystem::exists(output_dir)) {
                std::filesystem::create_directory(output_dir);  // Create directory if it doesn't exist
            }
            plotTexture.copyToImage().saveToFile(output_dir + "/" + filePath);
        }

        if (ImGui::Button("Close")) {
            window.close();
        }

        ImGui::End();
    }

    window.clear();
    ImGui::SFML::Render(window);
    window.display();
}

void PlotWindow::update() {
    sf::Event event;
    while (window.pollEvent(event)) {
        ImGui::SFML::ProcessEvent(window, event);

        if (event.type == sf::Event::Closed) {
            window.close();
        }
    }
}

bool PlotWindow::isOpen() const {
    return window.isOpen();
}

void PlotWindow::display() {
    window.display();
}