/* Copyright (C) 2016-2017 Alexandru-Valentin Musat (alexandruvalentinmusat@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <vector>
#include <string>
#include <boost/preprocessor.hpp>

#ifndef _NEXURALNET_UTILITY_EXCEPTIONS_H
#define _NEXURALNET_UTILITY_EXCEPTIONS_H

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
			(EmptyParams)
			(EmptyParamValue)
			(NoInputs)
			(ParamNotFound)
			(ParamParseException)
		)

			/*ExceptionType exception = strToEnum<ExceptionType>("EmptyImage");
			std::string exceptionStr = enumToStr(exception); */
	}
}
#endif
