package components.publik

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp


private val FluentPrimary = Color(0xFF0067C0)
private val FluentPrimaryLight = Color(0xFF60CDFF)


@Composable
fun PanelHeader(title: String, subtitle: String, isDarkTheme: Boolean = false) {
    Column {
        Text(
            text = subtitle.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(letterSpacing = 1.5.sp),
            color = if (isDarkTheme) FluentPrimaryLight else FluentPrimary.copy(alpha = 0.8f),
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = title,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = if (isDarkTheme) Color.White else MaterialTheme.colorScheme.onSurface
        )
    }
}