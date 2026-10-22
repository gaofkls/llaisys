// The half precision conversions used to live here. They are now `inline` in
// types.hpp so the ops kernels can inline (and vectorise) them; this translation
// unit is kept so the build layout stays unchanged.
#include "types.hpp"
