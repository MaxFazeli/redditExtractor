def parse_remote_host(host):
    """Splits "user@host[:port]" into ("user@host", port or None)."""
    user, _, rest = host.partition('@')
    hostname, _, port = rest.partition(':')
    return f"{user}@{hostname}", (port or None)


def ssh_prefix(host):
    """Returns the ssh command prefix for "user@host[:port]".

    BatchMode makes ssh fail immediately instead of waiting for a password prompt (there is no
    terminal inside the container); ConnectTimeout avoids hanging forever on network problems.
    """
    user_host, port = parse_remote_host(host)
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30']
    if port:
        command += ['-p', port]
    return command + [user_host]
