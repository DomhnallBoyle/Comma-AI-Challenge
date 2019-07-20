"""
    Filename: utils/telegram_bot_callback.py
    Description: Contains functionality for creating a Keras Callback for providing notifications via Telegram
    Author: Eyal Zakkay - Open-source taken from the Telegrad Project @ https://eyalzk.github.io/
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from keras.callbacks import Callback
import keras.backend as K

# local source imports
from dl_bot import DLBot


class TelegramBotCallback(Callback):
    """Callback that sends metrics and responds to Telegram Bot.

    Supports the following commands:
    /start: activate automatic updates every epoch
    /help: get a reply with all command options
    /status: get a reply with the latest epoch's results
    /getlr: get a reply with the current learning rate
    /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
    /plot: get a reply with the loss convergence plot image
    /quiet: stop getting automatic updates each epoch
    /stoptraining: kill Keras training process

    Attributes:
        kbot (DLBot): Instance of the DLBot class, holding the appropriate bot token

    Raises:
        TypeError: In case kbot is not a DLBot instance.
    """

    def __init__(self, kbot):
        """Instantiating an instance of TelegramBotCallback

        Calls the __init__ of the superclass Keras Callback

        Args:
            kbot (DLBot): an object for interacting with a Telegram bot to monitor and control the Keras training
            process
        """
        assert isinstance(kbot, DLBot), 'Bot must be an instance of the DLBot class'
        super(TelegramBotCallback, self).__init__()
        self.kbot = kbot

    def on_train_begin(self, logs=None):
        """Overridden function to be ran when the training process begins

        Activates the Telegram Bot and creates structures to record data

        Args:
            logs (Dictionary): containing the current training logs e.g. loss

        Returns:
            None
        """
        logs = {}

        # Add learning rate to logs dictionary
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        # Update bot's value of current LR
        self.kbot.lr = logs['lr']

        # Activate the telegram bot
        self.kbot.activate_bot()

        # set the number of epochs
        self.epochs = self.params['epochs']

        # loss history tracking
        self.loss_hist = []
        self.val_loss_hist = []

    def on_train_end(self, logs=None):
        """Overridden function to be ran when the training process ends.

        Notifies the user that the training has been completed and ends the bot

        Args:
            logs (Dictionary): containing the current training logs e.g. loss

        Returns:
            None
        """
        # send a message that the training has been completed and stop the bot
        self.kbot.send_message('Train Completed!')
        self.kbot.stop_bot()

    def on_epoch_begin(self, epoch, logs=None):
        """Overridden function to be ran at the beginning of an epoch

        Checks if the learning rate has been changed by the user and updates the learning rate
        before the new epoch begins

        Args:
            epoch (Integer): current epoch number
            logs (Dictionary): containing the current training logs e.g. loss

        Returns:
            None
        """
        # check if learning rate should be changed
        if self.kbot.modify_lr != 1:
            # check if the model has a learning rate attribute
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')

            # get current lr
            lr = float(K.get_value(self.model.optimizer.lr))

            # set new lr
            lr = lr * self.kbot.modify_lr
            K.set_value(self.model.optimizer.lr, lr)

            # Set multiplier back to 1
            self.kbot.modify_lr = 1

            # send notification message that lr has been changed
            message = '\nEpoch %05d: setting learning rate to %s.' % (epoch + 1, lr)
            self.kbot.send_message(message)

    def on_epoch_end(self, epoch, logs=None):
        """Overridden function to be ran at the end of an epoch

        Args:
            epoch (Integer): current epoch number
            logs (Dictionary): containing the current training logs e.g. loss

        Returns:
            None
        """
        logs = logs or {}

        # Did user invoke STOP command
        if self.kbot.stop_train_flag:
            self.model.stop_training = True
            self.kbot.send_message('Training Stopped!')
            print('Training Stopped! Stop command sent via Telegram bot.')

        # LR handling
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.kbot.lr = logs['lr']  # Update bot's value of current LR

        # Epoch message handling
        tlogs = ', '.join([k+': '+'{:.4f}'.format(v) for k, v in zip(logs.keys(), logs.values())])  # Clean logs string
        message = 'Epoch %d/%d \n' % (epoch + 1, self.epochs) + tlogs

        # Send epoch end logs
        if self.kbot.verbose:
            self.kbot.send_message(message)

        # Update status message
        self.kbot.set_status(message)

        # Loss tracking
        # Track loss to export as an image
        self.loss_hist.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_loss_hist.append(logs['val_loss'])

        # update the bots loss and validation history
        self.kbot.loss_hist = self.loss_hist
        self.kbot.val_loss_hist = self.val_loss_hist
