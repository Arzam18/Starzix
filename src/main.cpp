// clang-format off

#include "utils.hpp"
#include "position.hpp"
#include "uci.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "Starzix by zzzzz" << std::endl;

    #if defined(__AVX512F__) && defined(__AVX512BW__)
        std::cout << "Using avx512 (fastest)" << std::endl;
    #elif defined(__AVX2__)
        std::cout << "Using avx2 (fast)" << std::endl;
    #elif defined(__aarch64__) && defined(__ARM_NEON)
        std::cout << "Using arm64 NEON (fast)" << std::endl;
    #else
        std::cout << "Using scalar NNUE evaluation" << std::endl;
    #endif

    Position pos = START_POS;
    Searcher searcher = { };
    printTTSize(searcher.mTT);

    // If a command is passed in program args, run it and exit

    std::string command = "";

    for (size_t i = 1; i < static_cast<size_t>(argc); i++)
        command += " " + std::string(argv[i]);

    trim(command);

    if (command != "")
    {
        try {
            uci::runCommand(command, pos, searcher);
        }
        catch (...) {
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    // UCI loop
    while (true)
    {
        std::getline(std::cin, command);
        uci::runCommand(command, pos, searcher);
    }

    return EXIT_SUCCESS;
}
