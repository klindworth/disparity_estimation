#ifndef NEURAL_NETWORK_SETTINGS_H
#define NEURAL_NETWORK_SETTINGS_H

#include <cstddef>

namespace neural_network
{

struct training_settings {
	std::size_t epochs;
	std::size_t batch_size;
	std::size_t training_error_calculation; ///< output every n-th epoch the training error
};

}
#endif
