
// Copyright (C) 2025 Zac Sweers
// SPDX-License-Identifier: Apache-2.0

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.DependencyGraph
import dev.zacsweers.metro.Provides
import dev.zacsweers.metrox.viewmodel.ViewModelGraph

@DependencyGraph(AppScope::class) interface AppGraph : ViewModelGraph
{

}