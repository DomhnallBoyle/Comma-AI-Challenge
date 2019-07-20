"""
    Filename: utils/dl_bot.py
    Description: Contains functionality for interacting with Telegram to send training notifications to the user
    Author: Eyal Zakkay - Open-source taken from the Telegrad Project @ https://eyalzk.github.io/
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, Filters, RegexHandler,
                          ConversationHandler)
import numpy as np
import logging
from io import BytesIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class DLBot(object):
    """A class for interacting with a Telegram bot to monitor and control a Keras\tensorflow training process.

    Supports the following commands:
    /start: activate automatic updates every epoch and get a reply with all command options
    /help: get a reply with all command options
    /status: get a reply with the latest epoch's results
    /getlr: get a reply with the current learning rate
    /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
    /plot: get a reply with the loss convergence plot image
    /quiet: stop getting automatic updates each epoch
    /stoptraining: kill training process

    Attributes:
        token (String): a telegram bot API token
        user_id (Integer): Specifying a telegram user id will filter all incoming
                 commands to allow access only to a specific user. Optional, though highly recommended.
        filters (telegram.ext.filters.Filters): predefined filters for use as the filter argument of MessageHandler
        chat_id (Integer): the current Telegram chat ID
        bot_active (Boolean): indicates whether the Telegram Bot is active or not
        _status_message (String): status message that can be set
        lr (Float): the current learning rate of the training process
        modify_lr (Float): initial lr multiplier
        verbose (Boolean): for debugging purposes
        stop_train_flag (Boolean): flag that is set to stop the training process
        updater (telegram.ext.Updater): to receive the updates from Telegram and to deliver them to said dispatcher
        loss_hist (List): contains history of the training losses
        val_loss_hist (List): contains history of the validation losses
        logger (Object): for logging/debugging helpful messages
        startup_message (String): message to be sent when beginning training
        current_epoch (Integer): the current epoch of the training process
        current_loss (Float): the current training loss
        current_val_loss (Float): the current validation loss
        current_acc (Float): the current training accuracy
        current_val_acc (Float: the current validation accuracy
    """

    def __init__(self, token, user_id=None):
        """Instantiating an instance of DLBot

        Args:
            token (String): telegram API token
            user_id (Integer): ID to identify the user
        """
        assert isinstance(token, str), 'Token must be of type string'
        assert user_id is None or isinstance(user_id, int), 'user_id must be of type int (or None)'

        self.token = token  # bot token
        self.user_id = user_id  # id of the user with access
        self.filters = None
        self.chat_id = None  # chat id, will be fetched during /start command
        self.bot_active = False  # currently not in use
        self._status_message = "No status message was set"  # placeholder status message
        self.lr = None
        self.modify_lr = 1.0  # Initial lr multiplier
        self.verbose = True   # Automatic per epoch updates
        self.stop_train_flag = False  # Stop training flag
        self.updater = None
        # Initialize loss monitoring
        self.loss_hist = []
        self.val_loss_hist = []
        # Enable logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Message to display on /start and /help commands
        self.startup_message = "Hi, I'm the DL bot! I will send you updates on your training process.\n" \
                               " send /start to activate automatic updates every epoch\n" \
                               " send /help to see all options.\n" \
                               " Send /status to get the latest results.\n" \
                               " Send /getlr to query the current learning rate.\n" \
                               " Send /setlr to change the learning rate.\n" \
                               " Send /quiet to stop getting automatic updates each epoch\n" \
                               " Send /plot to get a loss convergence plot.\n" \
                               " Send /stoptraining to stop training process.\n\n"

        self.current_epoch = None
        self.current_loss = None
        self.current_val_loss = None
        self.current_acc = None
        self.current_val_acc = None

    def activate_bot(self):
        """Function to initiate the Telegram bot

        Returns:
            None
        """
        self.updater = Updater(self.token)  # setup updater
        dp = self.updater.dispatcher  # Get the dispatcher to register handlers
        dp.add_error_handler(self.error)  # log all errors

        # filter messages to allow only those which are from specified user ID.
        self.filters = Filters.user(user_id=self.user_id) if self.user_id else None

        # Command and conversation handles
        dp.add_handler(CommandHandler("start", self.start, filters=self.filters))  # /start
        dp.add_handler(CommandHandler("help", self.help, filters=self.filters))  # /help
        dp.add_handler(CommandHandler("status", self.status, filters=self.filters))  # /get status
        dp.add_handler(CommandHandler("getlr", self.get_lr, filters=self.filters))  # /get learning rate
        dp.add_handler(CommandHandler("quiet", self.quiet, filters=self.filters))  # /stop automatic updates
        dp.add_handler(CommandHandler("plot", self.plot_loss, filters=self.filters))  # /plot loss
        dp.add_handler(self.lr_handler())  # set learning rate
        dp.add_handler(self.stop_handler())  # stop training

        # Start the Bot
        self.updater.start_polling()
        self.bot_active = True

        # Uncomment next line while debugging
        # updater.idle()

    def stop_bot(self):
        """Function to stop the bot

        Returns:
            None
        """
        # stop the bot and change the active flag
        self.updater.stop()
        self.bot_active = False

    def start(self, bot, update):
        """Telegram bot callback for the /start command.

        Fetches chat_id, activates automatic epoch updates and sends startup message

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # send the startup message
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())

        # update the chat id from the message
        self.chat_id = update.message.chat_id

        # make the bot verbose - sends updates after every epoch
        self.verbose = True

    def help(self, bot, update):
        """Telegram bot callback for the /help command. Replies the startup message

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # send the startup message containing the helpful information
        update.message.reply_text(self.startup_message, reply_markup=ReplyKeyboardRemove())

        # update the chat id from the message
        self.chat_id = update.message.chat_id

    def quiet(self, bot, update):
        """Telegram bot callback for the /quiet command. Stops automatic epoch updates

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # turn off verbose updataes - only sent when you send a message handler
        self.verbose = False

        # send the reply message
        update.message.reply_text("Automatic epoch updates turned off. Send /start to turn epoch updates back on.")

    def error(self, update, error):
        """Log Errors caused by Updates.

        Returns:
            None
        """
        # send a warning to the logger caused by an update
        self.logger.warning('Update "%s" caused error "%s"', update, error)

    def send_message(self, txt):
        """Function to send a Telegram message to user

        Args:
            txt (String): the message to be sent

        Returns:
            None
        """
        # ensure the text if of the appropriate type
        assert isinstance(txt, str), 'Message text must be of type string'

        # send the text message through the updater if the chat id is known
        if self.chat_id is not None:
            self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
        else:
            # failed to send the message because the chat id is not known
            print('Send message failed, user did not send /start')

    def set_status(self, txt):
        """Function to set a status message to be returned by the /status command

        Args:
            txt (String): status message

        Returns:
            None
        """
        # ensure the status message is of type string before updating the status message
        assert isinstance(txt, str), 'Status Message must be of type string'
        self._status_message = txt

    def status(self, bot, update):
        """Telegram bot callback for the /status command. Replies with the latest status

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # send the training and validation losses as the status message if they both contain losses
        # if not, it indicates that the first epoch has not finished
        if len(self.loss_hist) != 0 and len(self.val_loss_hist) != 0:
            message = 'Training loss: {}, Validation loss: {}'.format(self.loss_hist[:-1], self.val_loss_hist[:-1])
        else:
            message = 'First epoch not finished.'

        # send the reply to through the updater
        update.message.reply_text(message)

    def get_lr(self, bot, update):
        """Telegram bot callback for the /getlr command. Replies with current learning rate

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # reply with the current learning rate if it exists
        if self.lr:
            update.message.reply_text("Current learning rate: " + str(self.lr))
        else:
            update.message.reply_text("Learning rate was not passed to DL-Bot")

    def set_lr_front(self, bot, update):
        """Telegram bot callback for the /setlr command. Displays option buttons for learning rate multipliers

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): indicating that the reply message has been sent
        """
        # possible multipliers
        reply_keyboard = [['X0.5', 'X0.1', 'X2', 'X10']]

        # show message with option buttons
        update.message.reply_text(
            'Change learning rate, multiply by a factor of: '
            '(Send /cancel to leave LR unchanged).\n\n',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard))

        return 1

    def set_lr_back(self, bot, update):
        """Telegram bot callback for the /setlr command. Handle user selection as part of conversation

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): a constant to return when a conversation is ended.
        """
        # possible multipliers
        options = {'X0.5': 0.5, 'X0.1': 0.1, 'X2': 2.0, 'X10': 10.0}

        # get the user selection
        self.modify_lr = options[update.message.text]

        # reply indicating that the learning rate will be modified by a factor specified
        update.message.reply_text(" Learning rate will be multiplied by {} on the beginning of next epoch!"
                                  .format(str(self.modify_lr)), reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_lr(self, bot, update):
        """Telegram bot callback for the /setlr command. Handle user cancellation as part of conversation

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): a constant to return when a conversation is ended.
        """
        # send a message indicating that the learning rate will not be modified
        self.modify_lr = 1.0
        update.message.reply_text('OK, learning rate will not be modified on next epoch.',
                                  reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def lr_handler(self):
        """Function to setup the callbacks for the /setlr command. Returns a conversation handler

        Returns:
            (ConversationHandler): A handler to hold a conversation with a single user by
            managing four collections of other handlers
        """
        # create a conversational handler that branches off into multiple handles depending on the action of the user
        # send the reply in the entry point, set the learning rate if necessary or cancel the operation
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('setlr', self.set_lr_front, filters=self.filters)],
            states={1: [RegexHandler('^(X0.5|X0.1|X2|X10)$', self.set_lr_back)]},
            fallbacks=[CommandHandler('cancel', self.cancel_lr, filters=self.filters)])

        return conv_handler

    def stop_training(self, bot, update):
        """Telegram bot callback for the /stoptraining command. Displays verification message with buttons

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): a constant indicating the reply message has been sent
        """
        # if the /stoptraining command is sent, reply with "Are you sure?" Yes/No options
        reply_keyboard = [['Yes', 'No']]
        update.message.reply_text(
                    'Are you sure? '
                    'This will stop your training process!\n\n',
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard))

        return 1

    def stop_training_verify(self, bot, update):
        """ Telegram bot callback for the /stoptraining command.

        Handle user selection as part of conversation for stopping the training process

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): a constant to return when a conversation is ended.
        """
        # get response
        is_sure = update.message.text

        # set the stop training flag depending on the response from the user
        # this flag is checked at the end of every epoch to see if training should continue
        # reply with the appropriate message
        if is_sure == 'Yes':
            self.stop_train_flag = True
            update.message.reply_text('OK, stopping training!', reply_markup=ReplyKeyboardRemove())
        elif is_sure == 'No':
            self.stop_train_flag = False  # to allow changing your mind before stop took place
            update.message.reply_text('OK, canceling stop request!', reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def cancel_stop(self, bot, update):
        """Telegram bot callback for the /stoptraining command.

        Handle user cancellation as part of conversation

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            (Integer): a constant to return when a conversation is ended.
        """
        # reply if the user cancels the stop operation
        self.stop_train_flag = False
        update.message.reply_text('OK, training will not be stopped.', reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def stop_handler(self):
        """Function to setup the callbacks for the /stoptraining command.

        Returns a conversation handler

        Returns:
            (telegram.ext.ConversationHandler): A handler to hold a conversation with a single user by managing
            four collections of other handlers
        """
        # create conversational handler to interact with the user and ask them if they really want to stop training
        # the stop_training_verify hander is called if the use replies, else cancel the operation
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('stoptraining', self.stop_training, filters=self.filters)],
            states={1: [RegexHandler('^(Yes|No)$', self.stop_training_verify)]},
            fallbacks=[CommandHandler('cancel', self.cancel_stop, filters=self.filters)])

        return conv_handler

    def plot_loss(self, bot, update):
        """Telegram bot callback for the /plot command.

        Replies with the current convergence plot image

        Args:
            bot (telegram.Bot): this object represents a Telegram Bot.
            update (telegram.Update): this object represents an incoming update.

        Returns:
            None
        """
        # send info message if first epoch wasn't finished or matplotlib isn't installed
        if not self.loss_hist or plt is None:
            update.message.reply_text('First epoch not finished or matplotlib not installed.')
            return

        loss_np = np.asarray(self.loss_hist)

        # Check if training has a validation set
        val_loss_np = np.asarray(self.val_loss_hist) if self.val_loss_hist else None

        # create the legend keys depending on the existence of the validation loss
        legend_keys = ['loss', 'val_loss'] if self.val_loss_hist else ['loss']

        # epoch axes
        x = np.arange(len(loss_np))
        fig = plt.figure()
        ax = plt.axes()

        # plot training loss
        ax.plot(x, loss_np, 'b')

        # plot val loss
        if val_loss_np is not None:
            ax.plot(x, val_loss_np, 'r')

        # add title, axis labels and legend to the plot
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax.legend(legend_keys)

        # send the image as a pack of bytes
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        update.message.reply_photo(buffer)
