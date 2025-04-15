#pragma once

#ifdef KERNELNET_EXPORTS
#define KERNELNET_API __declspec(dllexport)
#else
#define KERNELNET_API __declspec(dllimport)
#endif