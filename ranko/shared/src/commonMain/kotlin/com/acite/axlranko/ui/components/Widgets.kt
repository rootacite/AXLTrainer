package com.acite.axlranko.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextFieldColors
import androidx.compose.runtime.Composable
import androidx.compose.material3.LocalContentColor
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.acite.axlranko.ui.theme.rankoColors
import com.acite.axlranko.ui.theme.rankoTokens

@Composable
fun rankoFieldColors(): TextFieldColors {
    val colors = rankoColors
    return OutlinedTextFieldDefaults.colors(
        focusedBorderColor = colors.accentPink,
        unfocusedBorderColor = Color.White.copy(alpha = 0.10f),
        disabledBorderColor = Color.White.copy(alpha = 0.06f),
        errorBorderColor = colors.qualityRed,
        focusedLabelColor = colors.accentLilac,
        unfocusedLabelColor = colors.textDim,
        disabledLabelColor = colors.textDim.copy(alpha = 0.6f),
        errorLabelColor = colors.qualityRed,
        focusedTextColor = colors.text,
        unfocusedTextColor = colors.text,
        disabledTextColor = colors.textDim,
        errorTextColor = colors.text,
        cursorColor = colors.accentPink,
        focusedContainerColor = colors.bgCard.copy(alpha = 0.42f),
        unfocusedContainerColor = colors.bgCard.copy(alpha = 0.28f),
        disabledContainerColor = colors.bgCard.copy(alpha = 0.18f),
        errorContainerColor = colors.qualityRed.copy(alpha = 0.10f),
        focusedPlaceholderColor = colors.textDim,
        unfocusedPlaceholderColor = colors.textDim,
        focusedTrailingIconColor = colors.accentLilac,
        unfocusedTrailingIconColor = colors.textDim,
        focusedLeadingIconColor = colors.accentLilac,
        unfocusedLeadingIconColor = colors.textDim,
        focusedSupportingTextColor = colors.textDim,
        unfocusedSupportingTextColor = colors.textDim,
        errorSupportingTextColor = colors.qualityRed,
    )
}

@Composable
fun CapsuleButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    emphasized: Boolean = false,
    enabled: Boolean = true,
    danger: Boolean = false,
    compact: Boolean = false,
    contentPadding: PaddingValues = ButtonDefaults.ContentPadding,
    content: @Composable RowScope.() -> Unit = {
        Text(
            text,
            fontWeight = FontWeight.SemiBold,
            fontSize = if (compact) 12.sp else 14.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    },
) {
    val tokens = rankoTokens
    val colors = rankoColors
    val accent = if (danger) colors.qualityRed else colors.accentPink
    if (compact) {
        val bg = when {
            !enabled -> colors.bgCard.copy(alpha = 0.5f)
            emphasized || danger -> accent
            else -> colors.bgCard
        }
        val fg = when {
            !enabled -> colors.textDim
            danger -> Color.White
            emphasized -> colors.bgApp
            else -> colors.text
        }
        Box(
            modifier
                .height(36.dp)
                .clip(tokens.capsule)
                .background(bg)
                .clickable(enabled = enabled, onClick = onClick)
                .padding(horizontal = 12.dp),
            contentAlignment = Alignment.Center,
        ) {
            CompositionLocalProvider(LocalContentColor provides fg) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    content = content,
                )
            }
        }
        return
    }
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = tokens.capsule,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (emphasized || danger) accent else colors.bgCard,
            contentColor = when {
                danger -> Color.White
                emphasized -> colors.bgApp
                else -> colors.text
            },
            disabledContainerColor = colors.bgCard.copy(alpha = 0.5f),
            disabledContentColor = colors.textDim,
        ),
        contentPadding = contentPadding,
        modifier = modifier.height(52.dp),
        content = content,
    )
}

@Composable
fun CapsuleChoice(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val tokens = rankoTokens
    val colors = rankoColors
    val bg = if (selected) colors.accentPink.copy(alpha = 0.22f) else Color.White.copy(alpha = 0.05f)
    val border = if (selected) colors.accentPink.copy(alpha = 0.40f) else Color.White.copy(alpha = 0.10f)
    Box(
        modifier = modifier
            .height(40.dp)
            .clip(tokens.capsule)
            .background(bg)
            .border(1.dp, border, tokens.capsule)
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = colors.text, fontSize = 13.sp, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun RankoChoiceRow(
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    val tokens = rankoTokens
    val colors = rankoColors
    Row(
        modifier
            .alpha(if (enabled) 1f else 0.45f)
            .fillMaxWidth()
            .clip(tokens.panel)
            .background(
                if (selected) colors.accentPink.copy(alpha = 0.16f)
                else Color.White.copy(alpha = 0.05f),
            )
            .border(
                1.dp,
                if (selected) colors.accentPink.copy(alpha = 0.28f)
                else Color.White.copy(alpha = 0.08f),
                tokens.panel,
            )
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}

@Composable
fun QuietTextButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    val colors = rankoColors
    TextButton(onClick = onClick, enabled = enabled, modifier = modifier) {
        Text(text, color = colors.accentLilac, fontWeight = FontWeight.Medium)
    }
}
