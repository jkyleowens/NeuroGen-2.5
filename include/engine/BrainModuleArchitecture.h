#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <vector>
#include <string>
#include <memory>

// Forward declaration for learning state
struct LearningState;

/**
 * @brief Brain module architecture coordinator (stub for legacy compatibility)
 * 
 * This is a compatibility shim for the existing NetworkCUDA code.
 * The actual modular brain implementation is in modules/BrainOrchestrator.
 */
class BrainModuleArchitecture {
public:
    BrainModuleArchitecture() = default;
    virtual ~BrainModuleArchitecture() = default;
    
    /**
     * @brief Get number of modules
     */
    virtual size_t getModuleCount() const { return 0; }
    
    /**
     * @brief Get module names
     */
    virtual std::vector<std::string> getModuleNames() const { return {}; }
    
    /**
     * @brief Get global learning state
     */
    virtual LearningState getGlobalLearningState() const;
};

/**
 * @brief Learning state manager (stub for legacy compatibility)
 */
class LearningStateManager {
public:
    LearningStateManager() = default;
    virtual ~LearningStateManager() = default;
};

#endif // BRAIN_MODULE_ARCHITECTURE_H

