"""Independent public options for pose memory, channel gating, and state updates.

Compound configuration values are an internal adapter for the checkpoint-era
backbone; callers should use the independent options below.
"""


def add_memory_arguments(parser):
    group = parser.add_argument_group('Hybrid memory')
    group.add_argument('--pose_memory', choices=('ttt', 'local'), default='ttt',
                       help='Pose memory: fast weights (ttt) or CUT3R token memory (local). Requires matching checkpoint weights.')
    group.add_argument('--state_gate', choices=('channel', 'none'), default='channel',
                       help='Learned channel-wise geometry gate, or no learned gate.')
    group.add_argument('--state_update', choices=('direct', 'ttt3r', 'ttsa3r'), default='direct',
                       help='Geometry update rule, independent of the learned channel gate.')
    return parser


def model_memory_kwargs(args):
    """Translate independent choices for the checkpoint-compatible model loader."""
    update_modes = {
        ('channel', 'direct'): 'mem3r',
        ('channel', 'ttt3r'): 'mem3r_ttt3r',
        ('channel', 'ttsa3r'): 'mem3r_ttsa3r',
        ('none', 'direct'): 'cut3r',
        ('none', 'ttt3r'): 'ttt3r',
        ('none', 'ttsa3r'): 'ttsa3r',
    }
    return {'pose_retriever_type': args.pose_memory,
            'model_update_type': update_modes[(args.state_gate, args.state_update)]}


def configure_memory(model, args):
    """Record public settings and update the legacy backbone configuration."""
    model.config.pose_memory = args.pose_memory
    model.config.state_gate = args.state_gate
    model.config.state_update = args.state_update
    for name, value in model_memory_kwargs(args).items():
        setattr(model.config, name, value)
