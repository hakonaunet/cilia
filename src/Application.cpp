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
    } else if (mode == SimulationMode::Test1) {
        // Test1
        test1();
    }
    else if (mode == SimulationMode::Test2) {
        // Test2
        test2();
    }
}

void Application::run() {
    // Window setup
    sf::Vector2f windowsize(1350, 1350);
    int oscillatorSize = 30;
    sf::RenderWindow window(sf::VideoMode(windowsize.x, windowsize.y), "Simulation");
    window.setPosition(sf::Vector2i(3000, 0));

    // Second window setup
    sf::RenderWindow infoWindow(sf::VideoMode(windowsize.x, windowsize.y), "Velocity Field");
    infoWindow.setPosition(sf::Vector2i(1700, 0));

    // ImGui initialization
    if (!ImGui::SFML::Init(window)) {
        std::cerr << "Failed to initialize ImGui with SFML." << std::endl;
        return;  // Exit if initialization fails
    }

    sf::Clock deltaClock;

    // Unique pointers for grid and views
    std::unique_ptr<Grid> grid = nullptr;
    std::unique_ptr<Oseen> oseen = nullptr;
    std::unique_ptr<OscillatorView> kuramotoView = nullptr;
    std::unique_ptr<OseenView> oseenView = nullptr;

    // PlotWindow setup
    std::unique_ptr<PlotWindow> plotWindow = nullptr;

    while (window.isOpen()) {
        // Event handling
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed) window.close();
        }

        // Event handling for info window
        sf::Event infoEvent;
        while (infoWindow.pollEvent(infoEvent)) {
            if (infoEvent.type == sf::Event::Closed) infoWindow.close();
        }

        // Simulation start handling
        if (sharedData->startSimulation) {
            if (auto sharedDataKuramoto = dynamic_cast<SharedDataKuramoto*>(sharedData.get())) {
                grid = std::make_unique<Grid>(*sharedDataKuramoto);
                kuramotoView = std::make_unique<OscillatorView>(*grid, oscillatorSize);
            } else if (auto sharedDataOseen = dynamic_cast<SharedDataOseen*>(sharedData.get())) {
                oseen = std::make_unique<Oseen>(*sharedDataOseen);
                oseenView = std::make_unique<OseenView>(*oseen);
                oseenView->prepare(window);
            } else {
                std::cerr << "Error: sharedData could not be cast to a known type." << std::endl;
            }
            sharedData->startSimulation = false;
        }

        // Order parameter plotting
        if (sharedData->shouldPlotOrderParameter && grid != nullptr) {
            grid->plotOrderParameter("order_parameter.png");
            plotWindow = std::make_unique<PlotWindow>("order_parameter.png");
            sharedData->shouldPlotOrderParameter = false;
        }

        // PlotWindow update and display
        if (plotWindow && plotWindow->isOpen()) {
            plotWindow->update();
            plotWindow->display();
        } else {
            plotWindow.reset();
        }

        // Window clearing
        window.clear();
        infoWindow.clear();

        // Rendering
        if (sharedData->isSimulationRunning && (grid != nullptr || oseen != nullptr)) {
            if (kuramotoView) kuramotoView->renderSquares(window);
            else if (oseenView) oseenView->renderSquares(window);
        }

        // ImGui frame
        ImGui::SFML::Update(window, deltaClock.restart());
        ui->render();
        ImGui::SFML::Render(window);

        // Window display
        window.display();

        if (!sharedData->isPaused && (grid != nullptr || oseen != nullptr)) {
            if (dynamic_cast<SharedDataKuramoto*>(sharedData.get())) {
                grid->updateGrid();
            } else if (dynamic_cast<SharedDataOseen*>(sharedData.get())) {
                oseen->iteration();
            } else {
                std::cerr << "Error: sharedData could not be cast to a known type." << std::endl;
            }
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
        if (ImGui::Button("Oseen")) {
            mode = SimulationMode::Oseen;
            modeWindow.close();
        }
        if (ImGui::Button("Test1")) {
            ImGui::SetTooltip("Verify symmetry around the cilia plane");
            mode = SimulationMode::Test1;
            modeWindow.close();
        }
        if (ImGui::Button("Test2")) {
            ImGui::SetTooltip("Verify zero velocity in the z=0 plane");
            mode = SimulationMode::Test2;
            modeWindow.close();
        }
        ImGui::End();

        modeWindow.clear();
        ImGui::SFML::Render(modeWindow);
        modeWindow.display();
    }

    ImGui::SFML::Shutdown();
}