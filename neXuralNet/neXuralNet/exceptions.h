// Copyright (C) 2016 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

#include <vector>
#include <string>
#include <boost/preprocessor.hpp>

#ifndef MAVNET_UTILITY_EXCEPTIONS_H
#define MAVNET_UTILITY_EXCEPTIONS_H

namespace nexural {

#define X_STR_ENUM_TOSTRING_CASE(r, data, elem)                                 \
    case elem : return BOOST_PP_STRINGIZE(elem);

#define X_ENUM_STR_TOENUM_IF(r, data, elem)                                     \
    else if(data == BOOST_PP_STRINGIZE(elem)) return elem;

#define STR_ENUM(name, enumerators)                                             \
    enum name {                                                                 \
        BOOST_PP_SEQ_ENUM(enumerators)                                          \
    };                                                                          \
                                                                                \
    inline const std::string enumToStr(name v)                                      \
    {                                                                           \
        switch (v)                                                              \
        {                                                                       \
            BOOST_PP_SEQ_FOR_EACH(                                              \
                X_STR_ENUM_TOSTRING_CASE,                                       \
                name,                                                           \
                enumerators                                                     \
            )                                                                   \
                                                                                \
            default:                                                            \
                return "[Unknown " BOOST_PP_STRINGIZE(name) "]";                \
        }                                                                       \
    }                                                                           \
                                                                                \
    template <typename T>                                                       \
    inline const T strToEnum(std::string v);                                        \
                                                                                \
    template <>                                                                 \
    inline const name strToEnum(std::string v)                                      \
    {                                                                           \
        if(v=="")                                                               \
            throw std::runtime_error("Empty enum value");                       \
                                                                                \
        BOOST_PP_SEQ_FOR_EACH(                                                  \
            X_ENUM_STR_TOENUM_IF,                                               \
            v,                                                                  \
            enumerators                                                         \
        )                                                                       \
                                                                                \
        else                                                                    \
            throw std::runtime_error(                                           \
                        std::string("[Unknown value %1 for enum %2]")               \
                           );                        \
    }

	namespace mav_exception
	{
		STR_ENUM(
			ExceptionType,
			(EmptyImage)
			(EmptyMask)
			(EmptyTemplateCrop)
			(EmptyParams)
			(EmptyParamValue)
			(NoInputs)
			(ParamNotFound)
			(ParamParseException)
			(InvalidSigma)
			(InvalidVectorIndex)
			(InvalidImageDataType)
			(InvalidImageNbOfChannels)
			(InvalidClipLimit)
			(InvalidPercentage)
			(InvalidConversionCode)
			(InvalidRectangle)
			(InvalidHysteresisThresh)
			(InvalidApertureSize)
			(InvalidSize)
			(InvalidLevel)
			(InvalidApproxPrecision)
			(InvalidDownsize)
			(InvalidScalar)
			(InvalidLength)
			(InvalidAngleStep)
			(InvalidThickness)
			(InvalidRatio)
			(InvalidDelta)
			(InvalidMorphType)
			(InvalidMorphShape)
			(InvalidIterationsNb)
			(ImagesNotSameNbOfChannels)
			(ImagesNotSameSize)
			(ImagesNotSameDepth)
			(InvalidSuperpixelsNb)
			(InvalidLevelsNb)
			(InvalidThreshold)
			(InvalidAdaptiveMethod)
			(InvalidImgDimension)
			(BlobsDiffFeaturesDimension)
			(EmptyCriteriaList)
			(CriteriaValuesNotSameSize)
			(EmptyBlobCollection)
			(InvalidClustNb)
			(NoConsideredFeatures)
			(InvalidBlob)
			(InvalidInputData)
		)

			/*ExceptionType exception = strToEnum<ExceptionType>("EmptyImage");
			std::string exception_str = enumToStr(exception); */
	}
}

#endif

