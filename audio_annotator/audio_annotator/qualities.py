qualities = ('bright', 'dark', 'full', 'hollow', 'smooth', 'rough', 'warm', 'metallic',
             'clear', 'muddy', 'thin', 'thick', 'pure', 'noisy', 'rich', 'sparse',
             'harmonic', 'disharmonic', 'soft', 'hard')

pairs = [(qualities[i], qualities[i + 1]) for i in range(0, len(qualities), 2)]
