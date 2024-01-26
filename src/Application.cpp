// Application.cpp

#include "Application.hpp"

Application::Application() {
    promptForSimulationMode();
    init();
}

void Application::init() {
    if (mode == SimulationMode::Kuramoto) {
        sharedData = std::make_unique<SharedDataKuramoto>();
    } else if (mode == SimulationMode::Oseen) {
        sharedData = std::make_unique<SharedDataOseen>();
    }

    if (mode == SimulationMode::Kuramoto) {
        // Application.cpp
        auto sharedDataKuramoto = dynamic_cast<SharedDataKuramoto*>(sharedData.get());
        if (sharedDataKuramoto != nullptr) {
            grid = std::unique_ptr<Grid>(new Grid(*sharedDataKuramoto));
            ui = std::unique_ptr<UI>(new UI(*sharedDataKuramoto));
        } else {
            // Handle the error
        }
    } else if (mode == SimulationMode::Oseen) {
        auto sharedDataOseen = dynamic_cast<SharedDataOseen*>(sharedData.get());
        if (sharedDataOseen != nullptr) {
            oseen = std::make_unique<Oseen>(*sharedDataOseen);
            ui = std::make_unique<UI>(*sharedDataOseen);
        } else {
            // Handle the error
        }
    }
}

void Application::run() {
    sf::Vector2f windowsize(1350, 1350);
    int oscillatorSize = 30;
    sf::RenderWindow window(sf::VideoMode(windowsize.x, windowsize.y), "Kuramoto Simulation");

    // Set the position of the window (example: top-left corner of the screen)
    window.setPosition(sf::Vector2i(7500, 0));
    
    bool ImGui_initialization = ImGui::SFML::Init(window);
    if (!ImGui_initialization) {
        // Handle initialization failure
        std::cerr << "Failed to initialize ImGui with SFML." << std::endl;
    }

    sf::Clock deltaClock;
    std::unique_ptr<Grid> grid = nullptr;
    std::unique_ptr<OscillatorView> view = nullptr;

    PlotWindow* plotWindow = nullptr;


    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);

            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Check if the simulation should be started
        if (sharedData->isSimulationRunning && grid == nullptr) {
            auto sharedDataKuramoto = dynamic_cast<SharedDataKuramoto*>(sharedData.get());
            if (sharedDataKuramoto != nullptr) {
                grid = std::unique_ptr<Grid>(new Grid(*sharedDataKuramoto));
                view = std::make_unique<OscillatorView>(*grid, oscillatorSize);
                sharedData->startSimulation = false;
            } else {
                // Handle the error
            }
        }
        if (sharedData->startSimulation) {
            auto sharedDataKuramoto = dynamic_cast<SharedDataKuramoto*>(sharedData.get());
            if (sharedDataKuramoto != nullptr) {
                grid = std::unique_ptr<Grid>(new Grid(*sharedDataKuramoto));
                view = std::make_unique<OscillatorView>(*grid, oscillatorSize);
                sharedData.reset();
                sharedData->startSimulation = false;
            } else {
                // Handle the error
            }
        }

        // Check if the order parameter should be plotted
        if (sharedData->shouldPlotOrderParameter) {
            grid->plotOrderParameter("order_parameter.png");
            plotWindow = new PlotWindow("order_parameter.png");
            sharedData->shouldPlotOrderParameter = false;
        }
        // Update and display the plot window if it exists
        if (plotWindow != nullptr) {
            if (plotWindow->isOpen()) {
                plotWindow->update();  // Handle events and update the window
                plotWindow->display();  // Display the window
            } else {
                delete plotWindow;
                plotWindow = nullptr;
            }
        }

        // Clear the window
        window.clear();

        if (sharedData->isSimulationRunning && grid != nullptr) {
            view->renderSquares(window);
        }
        else if (sharedData->isSimulationRunning && oseen != nullptr) {
            view->renderCircles(window);
        }

        // Start the ImGui frame
        ImGui::SFML::Update(window, deltaClock.restart());
        ui->render();  // Assuming this method encapsulates ImGui render calls

        ImGui::SFML::Render(window);

        // Display the rendered frame
        window.display();

        if (!sharedData->isPaused && grid != nullptr) {
            grid->updateGrid();
        }
    }

    ImGui::SFML::Shutdown();
}

void Application::promptForSimulationMode() {
    sf::RenderWindow modeWindow(sf::VideoMode(200, 100), "Select Mode");
    bool initSuccess = ImGui::SFML::Init(modeWindow);
    if (!initSuccess) {
        // Handle initialization failure
        std::cerr << "Failed to initialize ImGui with SFML." << std::endl;
        return;
    }

    while (modeWindow.isOpen()) {
        sf::Event event;
        while (modeWindow.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(modeWindow, event);

            if (event.type == sf::Event::Closed)
                modeWindow.close();
        }

        ImGui::SFML::Update(modeWindow, deltaClock.restart());

        ImGui::Begin("Simulation Mode");
        if (ImGui::Button("Kuramoto")) {
            mode = SimulationMode::Kuramoto;
            modeWindow.close();
        }
        if (ImGui::Button("Mode 2")) {
            mode = SimulationMode::Oseen;
            modeWindow.close();
        }
        ImGui::End();

        modeWindow.clear();
        ImGui::SFML::Render(modeWindow);
        modeWindow.display();
    }

    ImGui::SFML::Shutdown();
}