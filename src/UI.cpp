// UI.cpp

#include "UI.hpp"

void UI::render() {
    ImGui::Begin("Simulation Controls");

    ImGui::SliderFloat("Time step", &sharedData->deltaTime, 0.001f, 1.0f);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Time step for simulation");
    }
    
    // Check if sharedData is of type SharedDataKuramoto
    SharedDataKuramoto* kuramotoData = dynamic_cast<SharedDataKuramoto*>(sharedData);
    if (kuramotoData != nullptr) {
        ImGui::SliderInt("Grid Size", &kuramotoData->N, 1, 1350);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Sidelength for the square grid of oscillators");
        }
        ImGui::SliderFloat("Epsilon", &kuramotoData->epsilon, 0.0f, 1.0f);
        if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("The width of the uniform distribution (centered around 1.0) of which each oscillator's intrinsic frequency is sampled from");
        }
        
        // ImGui drop down menu with options from the enum class 'Couplings'
        std::string couplingOptionsStr[] = { "None", "Nearest4", "Nearest8", "DistanceR", "DistanceR2", "Uniform" };
        const char* couplingOptions[sizeof(couplingOptionsStr) / sizeof(std::string)];
        for (size_t i = 0; i < sizeof(couplingOptionsStr) / sizeof(std::string); ++i) {
            couplingOptions[i] = couplingOptionsStr[i].c_str();
        }

        // ImGui dropdown menu to select coupling option
        if (ImGui::BeginCombo("Coupling", couplingOptions[static_cast<int>(kuramotoData->coupling)])) {
            for (size_t i = 0; i < sizeof(couplingOptions) / sizeof(const char*); ++i) {
                bool isSelected = (kuramotoData->coupling == static_cast<Coupling>(i));
                if (ImGui::Selectable(couplingOptions[i], isSelected)) {
                    kuramotoData->coupling = static_cast<Coupling>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    // Check if sharedData is of type SharedDataOseen
    SharedDataOseen* oseenData = dynamic_cast<SharedDataOseen*>(sharedData);
    if (oseenData != nullptr) {
        static bool independentHeight = false;
        ImGui::SliderInt("Width", &oseenData->width, 1, 400);
        ImGui::SameLine();
        ImGui::Checkbox("Height", &independentHeight);
        if (independentHeight) {
            ImGui::SliderInt("Height", &oseenData->height, 1, 400);
        } else {
            oseenData->height = oseenData->width;
        }
        ImGui::SliderFloat("Grid Spacing", &oseenData->gridSpacing, 1.0f, 10.0f);
        ImGui::SliderFloat("Cilia Radius", &oseenData->cilia_radius, 0.0f, 5.0f);
        ImGui::SliderFloat("Force", &oseenData->force, 0.0f, 10.0f);
        ImGui::SliderFloat("Force Amplitude", &oseenData->force_amplitude, 0.0f, oseenData->force);

        float itemWidth = ImGui::GetWindowWidth() / 8 - ImGui::GetStyle().ItemSpacing.x;
        ImGui::PushItemWidth(itemWidth);
        ImGui::InputFloat("Alfa1", &oseenData->alfa1, 0.0f, 0.0f, "%.4f");
        ImGui::SameLine();
        ImGui::InputFloat("Beta1", &oseenData->beta1, 0.0f, 0.0f, "%.4f");
        ImGui::SameLine();
        ImGui::InputFloat("Alfa2", &oseenData->alfa2, 0.0f, 0.0f, "%.4f");
        ImGui::SameLine();
        ImGui::InputFloat("Beta2", &oseenData->beta2, 0.0f, 0.0f, "%.4f");
        ImGui::PopItemWidth();

        // ImGui drop down menu with options from the enum class 'NoiseMode'
        std::string noiseModeOptionsStr[] = { "None", "Square", "Circular" };
        const char* noiseModeOptions[sizeof(noiseModeOptionsStr) / sizeof(std::string)];
        for (size_t i = 0; i < sizeof(noiseModeOptionsStr) / sizeof(std::string); ++i) {
            noiseModeOptions[i] = noiseModeOptionsStr[i].c_str();
        }

        // ImGui dropdown menu to select noise mode option
        if (ImGui::BeginCombo("Noise Mode", noiseModeOptions[static_cast<int>(oseenData->noiseMode)])) {
            for (size_t i = 0; i < sizeof(noiseModeOptions) / sizeof(const char*); ++i) {
                bool isSelected = (oseenData->noiseMode == static_cast<NoiseMode>(i));
                if (ImGui::Selectable(noiseModeOptions[i], isSelected)) {
                    oseenData->noiseMode = static_cast<NoiseMode>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // If noiseMode is not None, display a slider to manipulate noiseWidth
        if (oseenData->noiseMode != NoiseMode::None) {
            ImGui::SliderFloat("noiseWidth", &oseenData->noiseWidth, 0.0f, oseenData->noiseWidthMax);
        }

        // ImGui drop down menu with options from the enum class 'AngleDistribution'
        std::string angleDistributionOptionsStr[] = { "None", "Uniform", "Gaussian" };
        const char* angleDistributionOptions[sizeof(angleDistributionOptionsStr) / sizeof(std::string)];
        for (size_t i = 0; i < sizeof(angleDistributionOptionsStr) / sizeof(std::string); ++i) {
            angleDistributionOptions[i] = angleDistributionOptionsStr[i].c_str();
        }

        // ImGui dropdown menu to select angle distribution option
        if (ImGui::BeginCombo("Angle Distribution", angleDistributionOptions[static_cast<int>(oseenData->angleDistribution)])) {
            for (size_t i = 0; i < sizeof(angleDistributionOptions) / sizeof(const char*); ++i) {
                bool isSelected = (oseenData->angleDistribution == static_cast<AngleDistribution>(i));
                if (ImGui::Selectable(angleDistributionOptions[i], isSelected)) {
                    oseenData->angleDistribution = static_cast<AngleDistribution>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // If angleDistribution is Uniform, display a slider to manipulate angleWidth
        if (oseenData->angleDistribution == AngleDistribution::Uniform) {
            ImGui::SliderFloat("Angle Width", &oseenData->angleWidth, 0.0f, oseenData->angleWidthMax);
        }

        // If angleDistribution is Gaussian, display a slider to manipulate angleDeviation
        if (oseenData->angleDistribution == AngleDistribution::Gaussian) {
            ImGui::SliderFloat("Angle Deviation", &oseenData->angleDeviation, 0.0f, oseenData->angleDeviationMax);
        }
    }


    // Checkbox to set a new random seed
    static bool setRandomSeed = false;
    if (ImGui::Checkbox("Set Random Seed", &setRandomSeed)) {
        if (setRandomSeed) {
            srand(time(NULL)); // Set seed for random results
        } else {
            srand(0); // Set a fixed seed for consistent results
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enable to make the intrinsic frequencies of the oscillators different each time the simulation is run");
    }

    ImGui::SameLine(); // Display the following widgets on the same line

    // Button to set SharedDataKuramoto.trackKuramotoOrderParameter
    ImGui::Checkbox("Calculate Order Parameter", &sharedData->trackKuramotoOrderParameter);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enable to calculate and keep track of the Kuramoto order parameter");
    }

    ImGui::SameLine(); // Display the following widgets on the same line
    // Button to set SharedDataKuramoto.gpuParallelization
    ImGui::Checkbox("GPU parallelization", &sharedData->gpuParallelization);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enable to utilize GPU parallelization");
    }

    // Button to start the simulation
    if (ImGui::Button("Run Simulation")) {
        // Set flag to start simulation
        sharedData->isSimulationRunning = true;
        sharedData->startSimulation = true;
    }

    // Button to pause/resume the simulation
    if (!sharedData->isPaused) {
        if (ImGui::Button("Pause Simulation")) {
            pause();
        }
    }
    else {
        if (ImGui::Button("Resume Simulation")) {
            resume();
        }
    }

    // Display runtime information if simulation is running
    if (sharedData->isSimulationRunning) {
        ImGui::Text("Runtime: %s seconds", getRuntime().c_str());
        ImGui::Text("Simulation Time: %s seconds", getSimulationTime().c_str());
        ImGui::Text("Iteration: %d", sharedData->iteration);
        // Ensure that the UI updates are not too frequent
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsedTime = now - lastUpdateTime;

        static std::string updateTimeText = "Update Time: 0.000000 seconds";

        if (elapsedTime >= minUpdateInterval) {
            updateTimeText = "Update Time: " + std::to_string(sharedData->updateTime / 1e6) + " seconds";
            lastUpdateTime = now;
        }

        ImGui::Text("%s", updateTimeText.c_str());
    }

    if (ImGui::Button("Plot Order Parameter")) {
        sharedData->shouldPlotOrderParameter = true;
    }

    ImGui::End();

}

void UI::pause() {
    sharedData->isPaused = true;
    pauseStartTime = std::chrono::high_resolution_clock::now();
}

void UI::resume() {
    sharedData->isPaused = false;
    sharedData->totalPauseTime += std::chrono::high_resolution_clock::now() - pauseStartTime;
}

std::string UI::formatTime(int timeInSeconds) const {
    // Convert time to hours, minutes, and seconds
    int hours = timeInSeconds / 3600;
    int minutes = (timeInSeconds / 60) % 60;
    int seconds = timeInSeconds % 60;

    // Format the time
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << seconds;

    return ss.str();
}

std::string UI::getRuntime() const {
    static std::string lastFormattedTime; // Static variable to store the last returned string
    if (sharedData->isPaused) {
        return lastFormattedTime; // Return the last formatted time if simulation is paused
    }
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> runtime = currentTime - sharedData->startTime - sharedData->totalPauseTime;
    int runtimeInSeconds = int(runtime.count());
    lastFormattedTime = formatTime(runtimeInSeconds); // Update the last formatted time
    return lastFormattedTime;
}

std::string UI::getSimulationTime() const {
    int simulationTimeInSeconds = int(sharedData->simulationTime);
    return formatTime(simulationTimeInSeconds);
}