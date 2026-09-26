package com.acite.axlranko.util

import kotlin.math.absoluteValue
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt

fun formatFixed(value: Float, decimals: Int): String {
    if (decimals <= 0) return value.roundToInt().toString()
    val factor = 10.0.pow(decimals)
    val rounded = kotlin.math.round(value * factor) / factor
    val text = rounded.toString()
    val parts = text.split(".")
    val dec = (parts.getOrNull(1) ?: "").padEnd(decimals, '0').take(decimals)
    return "${parts[0]}.$dec"
}

fun formatFourDecimals(value: Float): String {
    val rounded = ((value * 10000).roundToInt() / 10000.0).toString()
    val parts = rounded.split(".")
    if (parts.size == 1) return "$rounded.0000"
    val decimalPart = parts[1].padEnd(4, '0')
    return "${parts[0]}.$decimalPart"
}

fun formatScientificTwoDecimals(value: Float): String {
    if (value == 0f) return "0.00e+00"
    val exponent = kotlin.math.floor(log10(value.absoluteValue).toDouble()).toInt()
    val mantissa = value / 10.0.pow(exponent)
    val roundedMantissa = ((mantissa * 100).roundToInt() / 100.0).toString()
    val parts = roundedMantissa.split(".")
    val formattedMantissa =
        if (parts.size == 1) "$roundedMantissa.00" else "${parts[0]}.${parts[1].padEnd(2, '0')}"
    val sign = if (exponent >= 0) "+" else "-"
    val formattedExponent = exponent.absoluteValue.toString().padStart(2, '0')
    return "${formattedMantissa}e$sign$formattedExponent"
}
