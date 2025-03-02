#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <iomanip>
#include "cuda_kernels.h"
#include "address_matcher.h"

// File Names
const std::string SETTINGS_FILE = "crypkey_settings.txt";
const std::string CHECKPOINT_FILE = "checkpoint.dat";
const std::string LOG_FILE = "crackbit.log";
const std::string RESULT_FILE = "found_keys.txt";

// Global State
std::atomic<bool> key_found(false);
std::mutex log_mutex;

// Puzzle Data (from btcpuzzle.info)
struct Puzzle {
    uint64_t start;
    uint64_t end;
    std::string address;
    double reward;
};

std::map<int, Puzzle> puzzles = {
    {68, {1ULL << 67, (1ULL << 68) - 1, "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", 6.80006914}},
    {69, {1ULL << 68, (1ULL << 69) - 1, "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", 6.90013661}},
    {71, {1ULL << 70, (1ULL << 71) - 1, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", 7.1000437}}
};

// Config Manager
class ConfigManager {
public:
    bool load() {
        std::ifstream file(SETTINGS_FILE);
        if (!file) return false;
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("PUZZLE=") == 0) puzzle_num = std::stoi(line.substr(7));
            else if (line.find("TOKEN=") == 0) token = line.substr(6);
            else if (line.find("WALLET=") == 0) wallet = line.substr(7);
            else if (line.find("GPUS=") == 0) gpus = std::stoi(line.substr(5));
        }
        file.close();
        return puzzles.count(puzzle_num) && gpus > 0;
    }

    void setup() {
        std::cout << "Welcome to CrypKey! First-time setup:\n";
        std::cout << "Choose puzzle (68, 69, 71): ";
        std::cin >> puzzle_num;
        while (!puzzles.count(puzzle_num)) {
            std::cout << "Invalid puzzle! Choose 68, 69, or 71: ";
            std::cin >> puzzle_num;
        }
        std::cout << "Puzzle #" << puzzle_num << ": " << puzzles[puzzle_num].reward 
                  << " BTC, Address: " << puzzles[puzzle_num].address << "\n";
        std::cout << "Enter user token (e.g., xyz123): ";
        std::cin >> token;
        std::cout << "Enter wallet address (e.g., 1A1zP1...): ";
        std::cin >> wallet;
        int detected_gpus;
        cudaGetDeviceCount(&detected_gpus);
        std::cout << "Detected " << detected_gpus << " GPUs. Use how many? (1-" << detected_gpus << "): ";
        std::cin >> gpus;
        gpus = std::min(std::max(1, gpus), detected_gpus);

        std::ofstream file(SETTINGS_FILE);
        file << "# CrypKey Settings\n";
        file << "PUZZLE=" << puzzle_num << " # Puzzle number (68, 69, 71)\n";
        file << "TOKEN=" << token << " # User token for credit\n";
        file << "WALLET=" << wallet << " # Bitcoin wallet address\n";
        file << "GPUS=" << gpus << " # Number of GPUs to use\n";
        file.close();
        std::cout << "Settings saved to " << SETTINGS_FILE << "\n";
    }

    int puzzle_num = 68;
    std::string token = "default_token";
    std::string wallet = "1DefaultWalletAddress";
    int gpus = 1;
};

// Progress Manager
class ProgressManager {
public:
    void load_checkpoint(int puzzle_num, int range) {
        std::ifstream file(CHECKPOINT_FILE, std::ios::binary);
        if (file) {
            file.read(reinterpret_cast<char*>(&last_key), sizeof(last_key));
            file.close();
        } else {
            last_key = calculate_start_key(puzzle_num, range);
        }
        start_key = calculate_start_key(puzzle_num, range);
        total_keys = calculate_range_size(puzzle_num);
    }

    void save_checkpoint() {
        std::ofstream file(CHECKPOINT_FILE, std::ios::binary);
        uint64_t processed = last_key + get_progress();
        file.write(reinterpret_cast<const char*>(&processed), sizeof(processed));
        file.close();
    }

    void get_user_range() {
        std::cout << "Enter starting 2-digit range (00–99): ";
        std::cin >> range;
        if (range < 0 || range > 99) {
            std::cout << "Invalid range! Defaulting to 00.\n";
            range = 0;
        }
    }

    void show_progress(int puzzle_num, int range) {
        auto start_time = std::chrono::steady_clock::now();
        while (!key_found) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            uint64_t processed = get_progress();
            float progress = (processed * 100.0f) / total_keys;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            float speed = elapsed > 0 ? (processed / 1e6) / elapsed : 0;
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "\r[Puzzle #" << puzzle_num << " | Range " << std::setw(2) << std::setfill('0') << range 
                      << "] Progress: " << std::fixed << std::setprecision(2) << progress 
                      << "% | Speed: " << speed << " MKeys/s" << std::flush;
        }
    }

    int range = 0;
    uint64_t start_key = 0;
private:
    uint64_t last_key = 0;
    uint64_t total_keys = 0;

    uint64_t calculate_start_key(int puzzle_num, int range) {
        uint64_t puzzle_start = puzzles[puzzle_num].start;
        uint64_t range_size = (puzzles[puzzle_num].end - puzzle_start + 1) / 100;
        return puzzle_start + range * range_size;
    }

    uint64_t calculate_range_size(int puzzle_num) {
        return (puzzles[puzzle_num].end - puzzles[puzzle_num].start + 1) / 100;
    }
};

// Logger
class Logger {
public:
    static void log(const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream file(LOG_FILE, std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        file << std::put_time(std::localtime(&now), "%F %T") << " - " << message << "\n";
    }

    static void save_result(const std::string& key, const std::string& token, const std::string& wallet) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream file(RESULT_FILE, std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        file << "Private Key: " << key << "\n";
        file << "Token: " << token << "\n";
        file << "Wallet: " << wallet << "\n";
        file << "Found: " << std::put_time(std::localtime(&now), "%F %T") << "\n\n";
    }
};

// Main Application
class CrypKey {
public:
    void run() {
        Logger::log("Application started");

        ConfigManager config;
        if (!config.load()) config.setup();

        ProgressManager progress;
        progress.get_user_range();
        progress.load_checkpoint(config.puzzle_num, progress.range);

        AddressMatcher::init(puzzles[config.puzzle_num].address.c_str());

        std::cout << "Targeting: " << puzzles[config.puzzle_num].address << "\n";
        std::cout << "Token: " << config.token << "\n";
        std::cout << "Wallet: " << config.wallet << "\n";
        std::cout << "Using " << config.gpus << " GPU(s)\n";

        std::thread progress_thread(&ProgressManager::show_progress, &progress, config.puzzle_num, progress.range);

        launch_cuda_kernel(progress.start_key, progress.total_keys, config.gpus);

        progress_thread.join();

        uint8_t private_key[32];
        if (check_results(private_key)) {
            std::stringstream ss;
            for (int i = 0; i < 32; i++) ss << std::hex << std::setw(2) << std::setfill('0') << (int)private_key[i];
            Logger::save_result(ss.str(), config.token, config.wallet);
            Logger::log("KEY FOUND! Application stopping");
            std::remove(CHECKPOINT_FILE.c_str());
        } else {
            Logger::log("No key found in range " + std::to_string(progress.range));
            std::cout << "\nNext: Range " << std::setw(2) << std::setfill('0') << (progress.range + 1) << "\n";
            progress.save_checkpoint();
        }

        reset_counters();
    }
};

int main() {
    try {
        CrypKey app;
        app.run();
    } catch (const std::exception& e) {
        Logger::log("CRITICAL ERROR: " + std::string(e.what()));
    }
    std::cout << "Press Enter to exit...\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}