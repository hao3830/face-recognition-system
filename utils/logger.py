import os
import datetime

from logging import handlers


class BiggerRotatingFileHandler(handlers.RotatingFileHandler):
    def __init__(
        self,
        alias,
        basedir,
        mode="a",
        maxBytes=0,
        backupCount=0,
        encoding=None,
        delay=0,
    ):
        """
        @summary:
        Set self.baseFilename to date string of today.
        The handler create logFile named self.baseFilename
        """
        self.basedir_ = basedir
        self.alias_ = alias

        self.baseFilename = self.getBaseFilename()

        handlers.RotatingFileHandler.__init__(
            self, self.baseFilename, mode, maxBytes, backupCount, encoding, delay
        )

    def getBaseFilename(self):
        """
        @summary: Return logFile name string formatted to "today.log.alias"
        """
        self.today_ = datetime.date.today()
        basename_ = self.today_.strftime("%Y-%m-%d") + "." + self.alias_ + ".log"
        return os.path.join(self.basedir_, basename_)

    def shouldRollover(self, record):
        """
        @summary:
        Rollover happen
        1. When the logFile size is get over maxBytes.
        2. When date is changed.

        @see: BaseRotatingHandler.emit
        """

        if self.stream is None:
            self.stream = self._open()

        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1

        if self.today_ != datetime.date.today():
            self.baseFilename = self.getBaseFilename()
            return 1

        return 0
