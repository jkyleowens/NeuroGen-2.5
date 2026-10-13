#include "PackedBackend.h"
#include "lif_spv.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace neurogen::cortical {
namespace {
void check(VkResult r) {
  if (r != VK_SUCCESS)
    throw std::runtime_error("Vulkan error " + std::to_string(r));
}
struct Buffer {
  VkDevice device;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  Buffer(VkDevice d, VkPhysicalDevice physical, size_t bytes) : device(d) {
    try {
      VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      info.size = bytes;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      check(vkCreateBuffer(device, &info, nullptr, &buffer));
      VkMemoryRequirements req{};
      vkGetBufferMemoryRequirements(device, buffer, &req);
      VkPhysicalDeviceMemoryProperties props{};
      vkGetPhysicalDeviceMemoryProperties(physical, &props);
      uint32_t type = UINT32_MAX;
      for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
          type = i;
          break;
        }
      if (type == UINT32_MAX)
        throw std::runtime_error(
            "Vulkan device lacks host-visible coherent storage memory");
      VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc.allocationSize = req.size;
      alloc.memoryTypeIndex = type;
      check(vkAllocateMemory(device, &alloc, nullptr, &memory));
      check(vkBindBufferMemory(device, buffer, memory, 0));
      check(vkMapMemory(device, memory, 0, bytes, 0, &mapped));
    } catch (...) {
      cleanup();
      throw;
    }
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void cleanup() {
    if (mapped)
      vkUnmapMemory(device, memory);
    if (buffer)
      vkDestroyBuffer(device, buffer, nullptr);
    if (memory)
      vkFreeMemory(device, memory, nullptr);
  }
  ~Buffer() { cleanup(); }
};
} // namespace
class VulkanBackend final : public PackedBackend {
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkShaderModule shader_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
  VkCommandPool commands_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkCommandBuffer command_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties props_{};
  std::unique_ptr<Buffer> input_, output_;
  size_t input_capacity_ = 0, output_capacity_ = 0;
  void cleanup() {
    if (device_)
      vkDeviceWaitIdle(device_);
    input_.reset();
    output_.reset();
    if (fence_)
      vkDestroyFence(device_, fence_, nullptr);
    if (commands_)
      vkDestroyCommandPool(device_, commands_, nullptr);
    if (descriptors_)
      vkDestroyDescriptorPool(device_, descriptors_, nullptr);
    if (pipeline_)
      vkDestroyPipeline(device_, pipeline_, nullptr);
    if (shader_)
      vkDestroyShaderModule(device_, shader_, nullptr);
    if (layout_)
      vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (descriptor_layout_)
      vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    if (device_)
      vkDestroyDevice(device_, nullptr);
    if (instance_)
      vkDestroyInstance(instance_, nullptr);
  }

public:
  VulkanBackend() {
    try {
      VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
      app.pApplicationName = "NeuroGen";
      app.apiVersion = VK_API_VERSION_1_0;
      VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      ic.pApplicationInfo = &app;
      check(vkCreateInstance(&ic, nullptr, &instance_));
      uint32_t count = 0;
      check(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
      std::vector<VkPhysicalDevice> devices(count);
      check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
      int selected = -1;
      if (const char *value = std::getenv("NEUROGEN_VULKAN_DEVICE")) {
        size_t consumed = 0;
        selected = std::stoi(value, &consumed);
        if (consumed != std::strlen(value) || selected < 0 ||
            selected >= static_cast<int>(count))
          throw std::runtime_error("Invalid NEUROGEN_VULKAN_DEVICE index");
      }
      uint32_t family = UINT32_MAX;
      int best = -1;
      for (uint32_t i = 0; i < count; ++i) {
        if (selected >= 0 && i != static_cast<uint32_t>(selected))
          continue;
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devices[i], &p);
        uint32_t queues = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queues, nullptr);
        std::vector<VkQueueFamilyProperties> families(queues);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queues,
                                                 families.data());
        for (uint32_t j = 0; j < queues; ++j)
          if (families[j].queueCount &&
              (families[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                        : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                            ? 2
                            : 1;
            if (score > best) {
              physical_ = devices[i];
              family = j;
              props_ = p;
              best = score;
            }
            break;
          }
      }
      if (!physical_)
        throw std::runtime_error("No Vulkan compute device available");
      if (props_.limits.maxComputeWorkGroupInvocations < 64 ||
          props_.limits.maxComputeWorkGroupSize[0] < 64)
        throw std::runtime_error("Vulkan workgroup size 64 unsupported");
      float priority = 1;
      VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      qi.queueFamilyIndex = family;
      qi.queueCount = 1;
      qi.pQueuePriorities = &priority;
      VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      di.queueCreateInfoCount = 1;
      di.pQueueCreateInfos = &qi;
      check(vkCreateDevice(physical_, &di, nullptr, &device_));
      vkGetDeviceQueue(device_, family, 0, &queue_);
      VkDescriptorSetLayoutBinding bindings[2]{};
      for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      }
      VkDescriptorSetLayoutCreateInfo dl{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      dl.bindingCount = 2;
      dl.pBindings = bindings;
      check(vkCreateDescriptorSetLayout(device_, &dl, nullptr,
                                        &descriptor_layout_));
      VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(Dispatch)};
      VkPipelineLayoutCreateInfo pl{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pl.setLayoutCount = 1;
      pl.pSetLayouts = &descriptor_layout_;
      pl.pushConstantRangeCount = 1;
      pl.pPushConstantRanges = &range;
      check(vkCreatePipelineLayout(device_, &pl, nullptr, &layout_));
      VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      si.codeSize = sizeof(lif_spv);
      si.pCode = lif_spv;
      check(vkCreateShaderModule(device_, &si, nullptr, &shader_));
      VkComputePipelineCreateInfo pi{
          VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pi.layout = layout_;
      pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pi.stage.module = shader_;
      pi.stage.pName = "main";
      check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr,
                                     &pipeline_));
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
      VkDescriptorPoolCreateInfo dp{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      dp.maxSets = 1;
      dp.poolSizeCount = 1;
      dp.pPoolSizes = &size;
      check(vkCreateDescriptorPool(device_, &dp, nullptr, &descriptors_));
      VkDescriptorSetAllocateInfo da{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      da.descriptorPool = descriptors_;
      da.descriptorSetCount = 1;
      da.pSetLayouts = &descriptor_layout_;
      check(vkAllocateDescriptorSets(device_, &da, &set_));
      VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      cp.queueFamilyIndex = family;
      cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      check(vkCreateCommandPool(device_, &cp, nullptr, &commands_));
      VkCommandBufferAllocateInfo ca{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      ca.commandPool = commands_;
      ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      ca.commandBufferCount = 1;
      check(vkAllocateCommandBuffers(device_, &ca, &command_));
      VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      check(vkCreateFence(device_, &fi, nullptr, &fence_));
    } catch (...) {
      cleanup();
      throw;
    }
  }
  ~VulkanBackend() { cleanup(); }
  std::string name() const override {
    return std::string("vulkan: ") + props_.deviceName +
           (props_.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? " (software)"
                                                             : "");
  }

private:
  Vector dispatch(const Vector &x, size_t n, Dispatch p) override {
    size_t bytes = x.size() * sizeof(float), outbytes = n * sizeof(float);
    if (bytes > props_.limits.maxStorageBufferRange ||
        outbytes > props_.limits.maxStorageBufferRange ||
        x.size() > UINT32_MAX || n > UINT32_MAX ||
        (p.n + 63) / 64 > props_.limits.maxComputeWorkGroupCount[0])
      throw std::runtime_error(
          "Model operation exceeds Vulkan device storage/dispatch limits");
    if (bytes > input_capacity_) {
      input_ = std::make_unique<Buffer>(device_, physical_, bytes);
      input_capacity_ = bytes;
    }
    if (outbytes > output_capacity_) {
      output_ = std::make_unique<Buffer>(device_, physical_, outbytes);
      output_capacity_ = outbytes;
    }
    auto &input = *input_;
    auto &output = *output_;
    std::memcpy(input.mapped, x.data(), bytes);
    VkDescriptorBufferInfo buffers[2] = {{input.buffer, 0, bytes},
                                         {output.buffer, 0, outbytes}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set_;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    check(vkResetCommandBuffer(command_, 0));
    check(vkResetFences(device_, 1, &fence_));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command_, &begin));
    vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                            0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(command_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(p), &p);
    vkCmdDispatch(command_, (p.n + 63) / 64, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                         0, nullptr);
    check(vkEndCommandBuffer(command_));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_));
    VkResult waited = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (waited != VK_SUCCESS) {
      vkDeviceWaitIdle(device_);
      check(waited);
    }
    Vector y(n);
    std::memcpy(y.data(), output.mapped, outbytes);
    return y;
  }
};
std::unique_ptr<ComputeBackend> make_vulkan() {
  return std::make_unique<VulkanBackend>();
}
} // namespace neurogen::cortical
